/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/ascend/transforms/aclnn_convolution_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/backends/ascend/transforms/aclnn_config.h"
#include "xla/backends/ascend/transforms/aclnn_targets.h"

namespace xla {
namespace ascend {
namespace {

namespace m = match;

using absl::StatusOr;

absl::Status SetName(HloModule* module, HloInstruction* instr) {
  module->SetAndUniquifyInstrName(instr, "aclnn-convolution");
  return absl::OkStatus();
}

bool IsBiasCompatibleWithOutput(const HloInstruction* bias, const HloInstruction* conv) {
  const Shape& bias_shape = bias->shape();
  const Shape& conv_shape = conv->shape();

  if (bias_shape.dimensions().size() == 1) {
    int64_t output_channel_dim = 3;
    if (conv_shape.dimensions().size() == 4) {
      return bias_shape.dimensions(0) == conv_shape.dimensions(output_channel_dim);
    }
  }
  return false;
}

bool IsAclnnSupportedConvolution(const HloConvolutionInstruction* conv) {
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& weight_shape = conv->operand(1)->shape();

  if (input_shape.dimensions().size() != 4 || weight_shape.dimensions().size() != 4) {
    VLOG(4) << "Only 4D convolutions are supported";
    return false;
  }
  
  PrimitiveType input_type = input_shape.element_type();
  if (input_type != F32 && input_type != F16 && input_type != BF16) {
    VLOG(4) << "Unsupported input type: " << PrimitiveType_Name(input_type);
    return false;
  }
  
  PrimitiveType weight_type = weight_shape.element_type();
  if (weight_type != F32 && weight_type != F16 && weight_type != BF16) {
    VLOG(4) << "Unsupported weight type: " << PrimitiveType_Name(weight_type);
    return false;
  }
  
  if (conv->feature_group_count() != 1) {
    VLOG(4) << "Group convolution not supported";
    return false;
  }

  const auto& dn = conv->convolution_dimension_numbers();
  if (dn.input_batch_dimension() != 0 || dn.output_batch_dimension() != 0) {
    VLOG(4) << "Batch dimension must be 0";
    return false;
  }
  
  for (int i = 0; i < conv->operand_count(); ++i) {
    if (conv->operand(i)->opcode() == HloOpcode::kReverse) {
      VLOG(4) << "Convolution with reverse operand is backward convolution, skipping";
      return false;
    }
  }
  
  return true;
}

AclnnConvolutionConfig ExtractConvolutionConfig(const HloConvolutionInstruction* conv) {
  AclnnConvolutionConfig config;

  const auto& window = conv->window();

  config.dilation.clear();
  config.output_padding.clear();
  
  for (size_t i = 0; i < window.dimensions().size(); ++i) {
    config.stride.push_back(window.dimensions(i).stride());
    config.dilation.push_back(window.dimensions(i).base_dilation());
    config.padding.push_back(window.dimensions(i).padding_low());
    config.padding.push_back(window.dimensions(i).padding_high());
  }

  bool has_reversal = false;
  for (size_t i = 0; i < window.dimensions().size(); ++i) {
    if (window.dimensions(i).window_reversal()) {
      has_reversal = true;
      break;
    }
  }
  config.transposed = has_reversal;
  
  for (size_t i = 0; i < config.stride.size(); ++i) {
    config.output_padding.push_back(0);
  }
  
  config.groups = conv->feature_group_count();
  config.cube_math_type = 0;
  config.has_bias = false;
  
  return config;
}

absl::StatusOr<bool> ProcessConvolution(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);
  
  // 跳过任何看起来像是反向卷积的卷积
  // 1. 检查是否有 reverse 操作数 - 这是输入梯度模式
  for (int i = 0; i < conv->operand_count(); ++i) {
    if (conv->operand(i)->opcode() == HloOpcode::kReverse) {
      return false;
    }
  }
  
  // 2. 检查维度标签是否是反向卷积的标签（权重梯度）
  const auto& dn = conv->convolution_dimension_numbers();
  std::string dim_labels = ConvolutionDimensionNumbersToString(dn);
  if (dim_labels.find("->01bf") != std::string::npos) {
    return false;
  }
  
  if (!IsAclnnSupportedConvolution(conv)) {
    return false;
  }
  
  std::vector<HloInstruction*> operands = {
      conv->mutable_operand(0),
      conv->mutable_operand(1)
  };
  
  AclnnConvolutionConfig config = ExtractConvolutionConfig(conv);
  config.has_bias = false;
  
  HloInstruction* conv_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(),
          operands,
          kAclnnConvolutionCallTarget));
  
  conv_call->set_raw_backend_config_string(config.ToString());
  TF_RETURN_IF_ERROR(SetName(instr->GetModule(), conv_call));
  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(conv_call));
  TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));
  
  return true;
}

class AclnnConvolutionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *bias = nullptr, *conv = nullptr;
    HloInstruction* optional_reshape = nullptr;
    HloInstruction* optional_broadcast = nullptr;

    if (Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithOpcode(HloOpcode::kConvolution).WithOneUser(),
                  m::Reshape(&optional_reshape,
                             m::Broadcast(&optional_broadcast,
                                          m::Parameter(&bias))))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithOpcode(HloOpcode::kConvolution).WithOneUser(),
                  m::Broadcast(&optional_broadcast,
                               m::Parameter(&bias)))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithOpcode(HloOpcode::kConvolution).WithOneUser(),
                  m::Reshape(&optional_reshape,
                             m::Parameter(&bias)))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithOpcode(HloOpcode::kConvolution).WithOneUser(),
                  m::Parameter(&bias)))) {

      if (!IsBiasCompatibleWithOutput(bias, conv)) {
        return absl::OkStatus();
      }

      std::vector<HloInstruction*> operands = {
          conv->mutable_operand(0),
          conv->mutable_operand(1),
          bias
      };

      AclnnConvolutionConfig config = ExtractConvolutionConfig(
          Cast<HloConvolutionInstruction>(conv));
      config.has_bias = true;

      HloInstruction* fused_conv = instr->AddInstruction(
          HloInstruction::CreateCustomCall(
              instr->shape(),
              operands,
              kAclnnConvolutionCallTarget));

      fused_conv->set_raw_backend_config_string(config.ToString());
      TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_conv));
      TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(fused_conv));
      TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));

      return absl::OkStatus();
    }

    if (Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithCustomCallTarget({kAclnnConvolutionCallTarget}),
                  m::Reshape(&optional_reshape,
                             m::Broadcast(&optional_broadcast,
                                          m::Parameter(&bias))))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithCustomCallTarget({kAclnnConvolutionCallTarget}),
                  m::Broadcast(&optional_broadcast,
                               m::Parameter(&bias)))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithCustomCallTarget({kAclnnConvolutionCallTarget}),
                  m::Reshape(&optional_reshape,
                             m::Parameter(&bias)))) ||
        Match(instr,
              m::AddAnyOrder(
                  m::Op(&conv).WithCustomCallTarget({kAclnnConvolutionCallTarget}),
                  m::Parameter(&bias)))) {

      if (!IsBiasCompatibleWithOutput(bias, conv)) {
        return absl::OkStatus();
      }

      TF_ASSIGN_OR_RETURN(auto config_or, ParseAclnnConfig(
          kAclnnConvolutionCallTarget, conv->raw_backend_config_string()));
      auto* conv_config = dynamic_cast<AclnnConvolutionConfig*>(config_or.get());
      if (!conv_config) {
        return absl::InternalError("Failed to cast to AclnnConvolutionConfig");
      }

      conv_config->has_bias = true;

      std::vector<HloInstruction*> operands(conv->operands().begin(),
                                            conv->operands().end());
      operands.push_back(bias);

      HloInstruction* fused_conv = instr->AddInstruction(
          HloInstruction::CreateCustomCall(
              instr->shape(),
              operands,
              kAclnnConvolutionCallTarget));

      fused_conv->set_raw_backend_config_string(conv_config->ToString());
      TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_conv));
      TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(fused_conv));
      TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));

      return absl::OkStatus();
    }

    return absl::OkStatus();
  }
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  
  // 收集需要处理的卷积指令
  std::vector<HloInstruction*> to_process;
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kConvolution) {
      to_process.push_back(instr);
    }
  }
  
  // 处理卷积指令
  for (auto* instr : to_process) {
    TF_ASSIGN_OR_RETURN(bool result, ProcessConvolution(instr));
    changed |= result;
  }
  
  // 使用 visitor 处理 bias fusion
  AclnnConvolutionRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  changed |= visitor.changed();
  
  return changed;
}

}  // anonymous namespace

absl::StatusOr<bool> AclnnConvolutionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace ascend
}  // namespace xla
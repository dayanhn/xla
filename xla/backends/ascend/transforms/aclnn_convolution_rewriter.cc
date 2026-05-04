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

bool IsBiasCompatibleWithOutput(const HloInstruction* bias,
                                const HloInstruction* conv) {
  const Shape& bias_shape = bias->shape();
  const Shape& conv_shape = conv->shape();

  if (bias_shape.dimensions().size() == 1) {
    int64_t output_feature_dim =
        conv->convolution_dimension_numbers().output_feature_dimension();
    if (output_feature_dim < conv_shape.dimensions().size()) {
      return bias_shape.dimensions(0) ==
             conv_shape.dimensions(output_feature_dim);
    }
  }
  return false;
}

bool IsAclnnSupportedConvolution(const HloConvolutionInstruction* conv) {
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& weight_shape = conv->operand(1)->shape();

  int64_t input_rank = input_shape.dimensions().size();
  int64_t weight_rank = weight_shape.dimensions().size();

  if (input_rank < 3 || input_rank > 5 ||
      weight_rank < 3 || weight_rank > 5) {
    VLOG(4) << "Only 1D/2D/3D convolutions are supported (rank 3-5), got "
            << "input_rank=" << input_rank << ", weight_rank=" << weight_rank;
    return false;
  }

  if (input_rank != weight_rank) {
    VLOG(4) << "Input and weight ranks must match";
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

  return true;
}

AclnnConvolutionConfig ExtractConvolutionConfig(
    const HloConvolutionInstruction* conv) {
  AclnnConvolutionConfig config;

  const auto& window = conv->window();

  config.dilation.clear();
  config.output_padding.clear();

  for (size_t i = 0; i < window.dimensions().size(); ++i) {
    config.stride.push_back(window.dimensions(i).stride());
    config.dilation.push_back(window.dimensions(i).window_dilation());
    config.padding.push_back(window.dimensions(i).padding_low());
    config.padding.push_back(window.dimensions(i).padding_high());
  }

  config.transposed = false;

  for (size_t i = 0; i < config.stride.size(); ++i) {
    config.output_padding.push_back(0);
  }

  config.groups = conv->feature_group_count();
  config.cube_math_type = 0;
  config.has_bias = false;

  const auto& dn = conv->convolution_dimension_numbers();
  config.dim_labels = ConvolutionDimensionNumbersToString(dn);

  return config;
}

absl::StatusOr<bool> ProcessConvolution(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);

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

  std::vector<HloInstruction*> to_process;
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kConvolution) {
      to_process.push_back(instr);
    }
  }

  for (auto* instr : to_process) {
    TF_ASSIGN_OR_RETURN(bool result, ProcessConvolution(instr));
    changed |= result;
  }

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

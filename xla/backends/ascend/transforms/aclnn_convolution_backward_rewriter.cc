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

#include "xla/backends/ascend/transforms/aclnn_convolution_backward_rewriter.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/ascend/transforms/aclnn_config.h"
#include "xla/backends/ascend/transforms/aclnn_targets.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace ascend {
namespace {

// 判断是否是反向卷积的输入梯度计算模式
// 特征：有一个操作数是 reverse 操作
bool IsGradInputConvolution(const HloConvolutionInstruction* conv) {
  VLOG(4) << "Checking IsGradInputConvolution for: " << conv->name();
  for (int i = 0; i < conv->operand_count(); ++i) {
    VLOG(4) << "  Operand " << i << ": " << conv->operand(i)->name() 
            << ", opcode: " << HloOpcodeString(conv->operand(i)->opcode());
    if (conv->operand(i)->opcode() == HloOpcode::kReverse) {
      VLOG(3) << "  Matched IsGradInputConvolution!";
      return true;
    }
  }
  return false;
}

// 判断是否是反向卷积的权重梯度计算模式
// 特征：维度标签包含 f01b_i01o->01bf
bool IsGradWeightConvolution(const HloConvolutionInstruction* conv) {
  VLOG(4) << "Checking IsGradWeightConvolution for: " << conv->name();
  const auto& dn = conv->convolution_dimension_numbers();
  std::string dim_labels = ConvolutionDimensionNumbersToString(dn);
  VLOG(4) << "  Dim labels: " << dim_labels;
  
  // 更精确的匹配：输出维度标签是 01bf
  bool matches = dim_labels.find("->01bf") != std::string::npos;
  if (matches) {
    VLOG(3) << "  Matched IsGradWeightConvolution!";
  }
  return matches;
}

// 提取卷积配置
void ExtractConvolutionConfig(const HloConvolutionInstruction* conv,
                              AclnnConvolutionBackwardConfig* config) {
  const Window& window = conv->window();
  
  // 提取 stride, padding, dilation
  config->stride.clear();
  config->padding.clear();
  config->dilation.clear();
  
  for (size_t i = 0; i < window.dimensions().size(); ++i) {
    config->stride.push_back(window.dimensions(i).stride());
    config->padding.push_back(window.dimensions(i).padding_low());
    config->padding.push_back(window.dimensions(i).padding_high());
    config->dilation.push_back(window.dimensions(i).base_dilation());
  }
  
  // 其他默认值
  config->transposed = false;
  config->output_padding = {0, 0};
  config->groups = 1;
  config->cube_math_type = 0;
}

// 处理输入梯度计算模式的卷积
absl::Status HandleGradInputConversion(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);
  VLOG(1) << "HandleGradInputConversion for: " << conv->name();
  
  // 找到 gradOutput 和原始 weight
  HloInstruction* grad_output = nullptr;
  HloInstruction* weight = nullptr;
  HloInstruction* reverse_weight = nullptr;
  
  for (int i = 0; i < conv->operand_count(); ++i) {
    auto* operand = conv->mutable_operand(i);
    VLOG(2) << "  Operand " << i << ": " << operand->name() 
            << ", opcode: " << HloOpcodeString(operand->opcode());
    if (operand->opcode() == HloOpcode::kReverse) {
      VLOG(2) << "    Found reverse operand!";
      reverse_weight = operand;
      weight = operand->mutable_operand(0);
      VLOG(2) << "    Original weight: " << weight->name();
    } else {
      grad_output = operand;
      VLOG(2) << "    Found grad_output: " << grad_output->name();
    }
  }
  
  if (!grad_output || !weight) {
    VLOG(1) << "  Missing grad_output or weight, skipping";
    return absl::OkStatus();
  }
  
  // 创建配置
  AclnnConvolutionBackwardConfig config;
  ExtractConvolutionConfig(conv, &config);
  config.output_mask = {true, false, false};  // 只计算 gradInput
  
  // 创建自定义调用
  std::vector<HloInstruction*> operands = {grad_output, weight};
  HloInstruction* custom_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(),
          operands,
          kAclnnConvolutionBackwardCallTarget));
  
  custom_call->set_raw_backend_config_string(config.ToString());
  
  custom_call->SetAndSanitizeName("aclnn-convolution-backward-grad-input");
  
  // 替换原指令
  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
  TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));
  
  // 如果 reverse_weight 现在没有用户了，也移除它
  if (reverse_weight && reverse_weight->user_count() == 0) {
    TF_RETURN_IF_ERROR(reverse_weight->parent()->RemoveInstruction(reverse_weight));
  }
  
  return absl::OkStatus();
}

// 处理权重梯度计算模式的卷积
absl::Status HandleGradWeightConvolution(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);
  VLOG(1) << "HandleGradWeightConvolution for: " << conv->name();
  
  // 获取两个操作数
  // 对于权重梯度卷积 f01b_i01o->01bf:
  // - operand(0) 是正向的输入 (input)
  // - operand(1) 是正向的输出梯度 (grad_output)
  HloInstruction* input = conv->mutable_operand(0);
  HloInstruction* grad_output = conv->mutable_operand(1);
  
  VLOG(2) << "  Input: " << input->name() << ", shape: " << input->shape().ToString();
  VLOG(2) << "  Grad output: " << grad_output->name() << ", shape: " << grad_output->shape().ToString();
  
  // 创建配置
  AclnnConvolutionBackwardConfig config;
  ExtractConvolutionConfig(conv, &config);
  
  // 对于权重梯度,需要设置 transposed=true
  // output_mask = {false, true, false} 表示只计算 gradWeight
  config.output_mask = {false, true, false};
  config.transposed = true;
  
  VLOG(2) << "  Config: transposed=" << config.transposed 
          << ", output_mask=[" << config.output_mask[0] << "," 
          << config.output_mask[1] << "," << config.output_mask[2] << "]";
  
  // aclnnConvolutionBackward 的参数顺序:
  // gradOutput, input, weight, biasSizes, stride, padding, dilation, ...
  // 对于权重梯度，需要传入 gradOutput, input, 和 weight（用于推断输出形状）
  // 从原始卷积的权重形状推断 weight 参数
  std::vector<HloInstruction*> operands = {grad_output, input};
  
  HloInstruction* custom_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(),
          operands,
          kAclnnConvolutionBackwardCallTarget));
  
  custom_call->set_raw_backend_config_string(config.ToString());
  
  custom_call->SetAndSanitizeName("aclnn-convolution-backward-grad-weight");
  
  // 替换原指令
  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
  TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));
  
  VLOG(1) << "  Successfully converted to aclnn-convolution-backward";
  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<bool> AclnnConvolutionBackwardRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> to_process;
    
    // 第一遍，收集需要处理的指令
    int conv_count = 0;
    for (auto* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConvolution) {
        conv_count++;
        auto* conv = Cast<HloConvolutionInstruction>(instr);
        
        bool is_grad_input = IsGradInputConvolution(conv);
        bool is_grad_weight = IsGradWeightConvolution(conv);
        
        if (is_grad_input || is_grad_weight) {
          to_process.push_back(instr);
        }
      }
    }
    
    // 第二遍，处理收集到的指令
    for (auto* instr : to_process) {
      auto* conv = Cast<HloConvolutionInstruction>(instr);
      
      if (IsGradInputConvolution(conv)) {
        TF_RETURN_IF_ERROR(HandleGradInputConversion(instr));
        changed = true;
      } else if (IsGradWeightConvolution(conv)) {
        TF_RETURN_IF_ERROR(HandleGradWeightConvolution(instr));
        changed = true;
      }
    }
  }
  
  return changed;
}

}  // namespace ascend
}  // namespace xla

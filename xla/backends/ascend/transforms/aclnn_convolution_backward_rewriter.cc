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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

  config->stride.clear();
  config->padding.clear();
  config->dilation.clear();

  for (size_t i = 0; i < window.dimensions().size(); ++i) {
    config->stride.push_back(window.dimensions(i).stride());
    config->padding.push_back(window.dimensions(i).padding_low());
    config->padding.push_back(window.dimensions(i).padding_high());
    config->dilation.push_back(window.dimensions(i).base_dilation());
  }

  config->transposed = false;
  config->output_padding = {0, 0};
  config->groups = 1;
  config->cube_math_type = 0;
}

// 检查两个卷积配置是否兼容（用于验证可融合）
bool AreConvConfigsCompatible(const AclnnConvolutionBackwardConfig& a,
                              const AclnnConvolutionBackwardConfig& b) {
  return a.stride == b.stride &&
         a.padding == b.padding &&
         a.dilation == b.dilation &&
         a.groups == b.groups;
}

// 判断是否是偏置梯度计算：reduce_sum(gradOutput) over batch+spatial dims
// 返回 true 并将 grad_output 设置为 traced-back source
// 同时遍历所有中间张量，检查它们是否匹配已知的 gradOutput
bool MatchGradBiasReduce(HloInstruction* instr,
                         HloInstruction** grad_output,
                         absl::flat_hash_set<HloInstruction*>* visited) {
  // 匹配 reduce 操作：对 gradOutput 在 batch 和 spatial 维度上求和
  if (instr->opcode() != HloOpcode::kReduce) {
    return false;
  }

  auto* reduce = Cast<HloReduceInstruction>(instr);
  if (reduce->dimensions().size() < 2) {
    return false;
  }

  // 检查 init value 是常量 0
  if (reduce->init_values().size() != 1) return false;
  auto* init = reduce->init_values()[0];
  if (init->opcode() != HloOpcode::kConstant) return false;
  if (!ShapeUtil::IsScalar(init->shape())) return false;

  // 检查计算是加法
  auto* reducer = reduce->called_computations()[0]->root_instruction();
  if (reducer->opcode() != HloOpcode::kAdd) return false;

  // 通过 reshapes、broadcasts 和 inner reduces 追溯操作数链，
  // 收集沿途访问到的所有指令
  HloInstruction* operand = reduce->mutable_operand(0);
  visited->insert(operand);

  while (operand->opcode() == HloOpcode::kReshape ||
         operand->opcode() == HloOpcode::kBroadcast) {
    operand = operand->mutable_operand(0);
    visited->insert(operand);
  }

  // 如果追溯到一个 reduce（双reduce模式：reduce→reshape→reduce）
  if (operand->opcode() == HloOpcode::kReduce) {
    auto* inner_reduce = Cast<HloReduceInstruction>(operand);
    operand = inner_reduce->mutable_operand(0);
    visited->insert(operand);
    while (operand->opcode() == HloOpcode::kReshape ||
           operand->opcode() == HloOpcode::kBroadcast) {
      operand = operand->mutable_operand(0);
      visited->insert(operand);
    }
  }

  *grad_output = operand;
  VLOG(3) << "Matched GradBias reduce: " << instr->name()
          << ", traced to grad_output: " << operand->name();
  return true;
}

// 一组共享相同 gradOutput 的反向卷积操作
struct ConvBackwardGroup {
  HloInstruction* grad_output = nullptr;
  HloInstruction* input = nullptr;             // 前向激活输入（用于gradWeight）
  HloInstruction* weight = nullptr;            // 前向权重
  HloInstruction* grad_input_conv = nullptr;   // 计算 gradInput 的卷积
  HloInstruction* grad_weight_conv = nullptr;  // 计算 gradWeight 的卷积
  HloInstruction* grad_bias_reduce = nullptr;  // 计算 gradBias 的 reduce
  HloInstruction* reverse_weight = nullptr;    // gradInput 里的 reverse(w)
  bool has_grad_input = false;
  bool has_grad_weight = false;
  bool has_grad_bias = false;
  AclnnConvolutionBackwardConfig config;
};

// 从 HLO 中收集所有反向卷积模式，并按 gradOutput 分组
void CollectBackwardGroups(
    HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, ConvBackwardGroup>* groups) {

  // 第一遍：寻找 gradInput 卷积（有 kReverse 操作数的卷积）
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kConvolution) continue;
    auto* conv = Cast<HloConvolutionInstruction>(instr);
    if (!IsGradInputConvolution(conv)) continue;

    HloInstruction* grad_output = nullptr;
    HloInstruction* weight = nullptr;
    HloInstruction* reverse_weight = nullptr;

    for (int i = 0; i < conv->operand_count(); ++i) {
      auto* operand = conv->mutable_operand(i);
      if (operand->opcode() == HloOpcode::kReverse) {
        reverse_weight = operand;
        weight = operand->mutable_operand(0);
      } else {
        grad_output = operand;
      }
    }

    if (!grad_output || !weight) continue;

    VLOG(2) << "Found gradInput conv: " << conv->name()
            << ", grad_output: " << grad_output->name()
            << ", weight: " << weight->name();

    auto& group = (*groups)[grad_output];
    group.grad_output = grad_output;
    group.grad_input_conv = instr;
    group.weight = weight;
    group.reverse_weight = reverse_weight;
    group.has_grad_input = true;
    ExtractConvolutionConfig(conv, &group.config);
  }

  // 第二遍：寻找 gradWeight 卷积（维度标签包含 ->01bf）
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kConvolution) continue;
    auto* conv = Cast<HloConvolutionInstruction>(instr);
    if (!IsGradWeightConvolution(conv)) continue;

    // gradWeight 卷积：operand(0) = 前向输入，operand(1) = gradOutput
    HloInstruction* input = conv->mutable_operand(0);
    HloInstruction* grad_output = conv->mutable_operand(1);

    VLOG(2) << "Found gradWeight conv: " << conv->name()
            << ", grad_output: " << grad_output->name()
            << ", input: " << input->name();

    auto& group = (*groups)[grad_output];
    group.grad_output = grad_output;
    group.grad_weight_conv = instr;
    group.input = input;
    group.has_grad_weight = true;

    if (!group.has_grad_input) {
      // 只有 gradWeight 时，从该卷积提取配置
      ExtractConvolutionConfig(conv, &group.config);
    } else {
      // 验证配置兼容性
      AclnnConvolutionBackwardConfig weight_config;
      ExtractConvolutionConfig(conv, &weight_config);
      if (!AreConvConfigsCompatible(group.config, weight_config)) {
        VLOG(1) << "Config mismatch for gradWeight conv " << conv->name()
                << " — skipping fusion with gradInput";
        group.has_grad_weight = false;
        group.grad_weight_conv = nullptr;
        group.input = nullptr;
      }
    }
  }

  // 第三遍：寻找偏置梯度 reduce 操作
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    HloInstruction* traced_grad_output = nullptr;
    absl::flat_hash_set<HloInstruction*> visited;
    if (!MatchGradBiasReduce(instr, &traced_grad_output, &visited)) continue;

    // 查找 traced_grad_output 或任何 visited 中间张量是否属于某个已知组
    // （因为 gradOutput 可能是 traced chain 中某个中间 reshape 的结果）
    auto it = groups->find(traced_grad_output);
    if (it == groups->end()) {
      for (auto* visited_instr : visited) {
        it = groups->find(visited_instr);
        if (it != groups->end()) break;
      }
    }
    if (it == groups->end()) continue;

    VLOG(2) << "Found gradBias reduce: " << instr->name()
            << ", matches grad_output: " << it->second.grad_output->name();

    auto& group = it->second;
    group.grad_bias_reduce = instr;
    group.has_grad_bias = true;
  }
}

// 处理融合的反向卷积操作：创建一个带 tuple 输出的 custom call
absl::Status HandleFusedConvBackward(ConvBackwardGroup* group) {
  if (!group->weight || !group->grad_output) {
    return absl::InternalError(
        "HandleFusedConvBackward: weight and grad_output must not be null");
  }
  VLOG(1) << "HandleFusedConvBackward: grad_output=" << group->grad_output->name()
          << ", has_grad_input=" << group->has_grad_input
          << ", has_grad_weight=" << group->has_grad_weight
          << ", has_grad_bias=" << group->has_grad_bias;

  auto* computation = group->grad_output->parent();

  // 设置 output_mask
  group->config.output_mask = {
      group->has_grad_input,
      group->has_grad_weight,
      group->has_grad_bias};

  // 构建操作数列表：[gradOutput, weight, input(optional)]
  std::vector<HloInstruction*> operands;
  operands.push_back(group->grad_output);
  operands.push_back(group->weight);
  if (group->has_grad_weight && group->input) {
    operands.push_back(group->input);
  }

  // 构建 tuple 输出 shape（按 gradInput, gradWeight, gradBias 顺序）
  std::vector<Shape> output_shapes;
  if (group->has_grad_input) {
    output_shapes.push_back(group->grad_input_conv->shape());
  }
  if (group->has_grad_weight) {
    output_shapes.push_back(group->grad_weight_conv->shape());
  }
  if (group->has_grad_bias) {
    output_shapes.push_back(group->grad_bias_reduce->shape());
  }

  Shape tuple_shape = ShapeUtil::MakeTupleShape(output_shapes);

  // 创建 fused custom call
  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          tuple_shape, operands, kAclnnConvolutionBackwardCallTarget));
  custom_call->set_raw_backend_config_string(group->config.ToString());
  custom_call->SetAndSanitizeName("aclnn-convolution-backward-fused");

  VLOG(2) << "Created fused custom call: " << custom_call->name()
          << ", shape: " << tuple_shape.ToString()
          << ", config: " << group->config.ToString();

  // 创建 GTE 并替换原始指令的用途
  int output_idx = 0;

  if (group->has_grad_input) {
    HloInstruction* gte = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            output_shapes[output_idx], custom_call, output_idx));
    gte->SetAndSanitizeName(
        absl::StrCat(custom_call->name(), ".grad_input"));
    TF_RETURN_IF_ERROR(group->grad_input_conv->ReplaceAllUsesWith(gte));
    output_idx++;
  }

  if (group->has_grad_weight) {
    HloInstruction* gte = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            output_shapes[output_idx], custom_call, output_idx));
    gte->SetAndSanitizeName(
        absl::StrCat(custom_call->name(), ".grad_weight"));
    TF_RETURN_IF_ERROR(group->grad_weight_conv->ReplaceAllUsesWith(gte));
    output_idx++;
  }

  if (group->has_grad_bias) {
    HloInstruction* gte = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            output_shapes[output_idx], custom_call, output_idx));
    gte->SetAndSanitizeName(
        absl::StrCat(custom_call->name(), ".grad_bias"));
    TF_RETURN_IF_ERROR(group->grad_bias_reduce->ReplaceAllUsesWith(gte));
    output_idx++;
  }

  // 移除原始指令（及其不再使用的操作数链）
  if (group->has_grad_input) {
    TF_RETURN_IF_ERROR(
        computation->RemoveInstruction(group->grad_input_conv));
    // 如果 reverse(weight) 现在没有用户，也移除它
    if (group->reverse_weight &&
        group->reverse_weight->user_count() == 0) {
      TF_RETURN_IF_ERROR(
          computation->RemoveInstruction(group->reverse_weight));
    }
  }

  if (group->has_grad_weight) {
    TF_RETURN_IF_ERROR(
        computation->RemoveInstruction(group->grad_weight_conv));
  }

  if (group->has_grad_bias) {
    // 使用 RemoveInstructionAndUnusedOperands 清理整条 reduce chain
    // （如 reduce->reshape->reduce->reshape->gradOutput），
    // 但 gradOutput 如果仍被 custom call 引用，则不会被移除
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(
            group->grad_bias_reduce));
  }

  return absl::OkStatus();
}

// 处理输入梯度计算模式的卷积（单操作数，未融合）
absl::Status HandleGradInputConversion(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);
  VLOG(1) << "HandleGradInputConversion for: " << conv->name();

  HloInstruction* grad_output = nullptr;
  HloInstruction* weight = nullptr;
  HloInstruction* reverse_weight = nullptr;

  for (int i = 0; i < conv->operand_count(); ++i) {
    auto* operand = conv->mutable_operand(i);
    if (operand->opcode() == HloOpcode::kReverse) {
      reverse_weight = operand;
      weight = operand->mutable_operand(0);
    } else {
      grad_output = operand;
    }
  }

  if (!grad_output || !weight) {
    return absl::OkStatus();
  }

  AclnnConvolutionBackwardConfig config;
  ExtractConvolutionConfig(conv, &config);
  config.output_mask = {true, false, false};

  std::vector<HloInstruction*> operands = {grad_output, weight};
  HloInstruction* custom_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(), operands,
          kAclnnConvolutionBackwardCallTarget));
  custom_call->set_raw_backend_config_string(config.ToString());
  custom_call->SetAndSanitizeName("aclnn-convolution-backward-grad-input");

  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
  TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));

  if (reverse_weight && reverse_weight->user_count() == 0) {
    TF_RETURN_IF_ERROR(
        reverse_weight->parent()->RemoveInstruction(reverse_weight));
  }

  return absl::OkStatus();
}

// 处理权重梯度计算模式的卷积（单操作数，未融合）
absl::Status HandleGradWeightConvolution(HloInstruction* instr) {
  auto* conv = Cast<HloConvolutionInstruction>(instr);
  VLOG(1) << "HandleGradWeightConvolution for: " << conv->name();

  HloInstruction* input = conv->mutable_operand(0);
  HloInstruction* grad_output = conv->mutable_operand(1);

  AclnnConvolutionBackwardConfig config;
  ExtractConvolutionConfig(conv, &config);
  config.output_mask = {false, true, false};
  config.transposed = true;

  std::vector<HloInstruction*> operands = {grad_output, input};

  HloInstruction* custom_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(), operands,
          kAclnnConvolutionBackwardCallTarget));
  custom_call->set_raw_backend_config_string(config.ToString());
  custom_call->SetAndSanitizeName("aclnn-convolution-backward-grad-weight");

  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
  TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(instr));

  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<bool> AclnnConvolutionBackwardRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {

    // 阶段 1：收集所有反向卷积模式并按 gradOutput 分组
    absl::flat_hash_map<HloInstruction*, ConvBackwardGroup> groups;
    CollectBackwardGroups(computation, &groups);

    // 阶段 2：处理融合组（同一个 gradOutput 上有2+操作）
    absl::flat_hash_set<HloInstruction*> processed;
    for (auto& [grad_output, group] : groups) {
      int count = (group.has_grad_input ? 1 : 0) +
                  (group.has_grad_weight ? 1 : 0) +
                  (group.has_grad_bias ? 1 : 0);

      // 只有 weight 可用（来自 gradInput）时才能融合
      // 没有 gradInput 时，无法获取 weight 张量，跳过融合
      bool can_fuse = count >= 2 && group.weight != nullptr;

      if (can_fuse) {
        VLOG(1) << "Fusing " << count << " backward ops for grad_output: "
                << grad_output->name();
        TF_RETURN_IF_ERROR(HandleFusedConvBackward(&group));
        changed = true;

        if (group.grad_input_conv) processed.insert(group.grad_input_conv);
        if (group.grad_weight_conv) processed.insert(group.grad_weight_conv);
        if (group.grad_bias_reduce) processed.insert(group.grad_bias_reduce);
      }
    }

    // 阶段 3：处理剩余的单独操作（保持现有行为）
    std::vector<HloInstruction*> to_process;
    for (auto* instr : computation->MakeInstructionPostOrder()) {
      if (processed.contains(instr)) continue;
      if (instr->opcode() != HloOpcode::kConvolution) continue;

      auto* conv = Cast<HloConvolutionInstruction>(instr);
      if (IsGradInputConvolution(conv) || IsGradWeightConvolution(conv)) {
        to_process.push_back(instr);
      }
    }

    for (auto* instr : to_process) {
      if (processed.contains(instr)) continue;
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

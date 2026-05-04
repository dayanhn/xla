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
#include <tuple>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/window_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace ascend {

using ConvolutionMatch = std::optional<
    std::tuple<Window, ConvolutionDimensionNumbers, HloInstruction*>>;

// Helper function to convert ConvolutionDimensionNumbers to dim_labels string
// dim_labels format: "input_weight->output"
// e.g., "bf01_oi01->bf01" for NCHW format
// For 4D (batch, feature, height, width):
//   NCHW: input=bf01, weight=oi01, output=bf01
//   NHWC: input=b01f, weight=oihw(or 01io), output=b01f
std::string ConvolutionDimensionNumbersToDimLabels(
    const ConvolutionDimensionNumbers& dnums,
    const Shape& input_shape,
    const Shape& weight_shape,
    const Shape& output_shape) {
  int num_spatial_dims = dnums.kernel_spatial_dimensions_size();

  auto build_input_output_label = [&](int64_t dim,
                                      int64_t batch_dim,
                                      int64_t feature_dim,
                                      const absl::Span<const int64_t> spatial_dims) -> char {
    if (dim == batch_dim) return 'b';
    if (dim == feature_dim) return 'f';
    for (int i = 0; i < spatial_dims.size(); ++i) {
      if (dim == spatial_dims[i]) return '0' + i;
    }
    return '?';
  };

  auto build_kernel_label = [&](int64_t dim,
                                int64_t output_feature_dim,
                                int64_t input_feature_dim,
                                const absl::Span<const int64_t> spatial_dims) -> char {
    if (dim == output_feature_dim) return 'o';
    if (dim == input_feature_dim) return 'i';
    for (int i = 0; i < spatial_dims.size(); ++i) {
      if (dim == spatial_dims[i]) return '0' + i;
    }
    return '?';
  };

  auto input_spatial = absl::MakeConstSpan(
      dnums.input_spatial_dimensions().data(),
      dnums.input_spatial_dimensions_size());
  auto kernel_spatial = absl::MakeConstSpan(
      dnums.kernel_spatial_dimensions().data(),
      dnums.kernel_spatial_dimensions_size());
  auto output_spatial = absl::MakeConstSpan(
      dnums.output_spatial_dimensions().data(),
      dnums.output_spatial_dimensions_size());

  std::string input_label;
  for (int64_t d = 0; d < input_shape.dimensions_size(); ++d) {
    input_label += build_input_output_label(d, dnums.input_batch_dimension(),
                                           dnums.input_feature_dimension(),
                                           input_spatial);
  }

  std::string weight_label;
  for (int64_t d = 0; d < weight_shape.dimensions_size(); ++d) {
    weight_label += build_kernel_label(d, dnums.kernel_output_feature_dimension(),
                                      dnums.kernel_input_feature_dimension(),
                                      kernel_spatial);
  }

  std::string output_label;
  for (int64_t d = 0; d < output_shape.dimensions_size(); ++d) {
    output_label += build_input_output_label(d, dnums.output_batch_dimension(),
                                            dnums.output_feature_dimension(),
                                            output_spatial);
  }

  return absl::StrCat(input_label, "_", weight_label, "->", output_label);
}

bool MaybeConv1dToConv2d(HloInstruction* conv) {
  if (conv->window().dimensions().size() != 2) {
    return false;
  }
  if (conv->operand(1)->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto filter = conv->operand(1);
  std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
      filter->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
  if (reshape_degenerate.has_value() &&
      reshape_degenerate->deleted_dimensions.empty() &&
      reshape_degenerate->inserted_dimensions.size() == 1) {
    const auto& dnums = conv->convolution_dimension_numbers();
    for (auto dim : dnums.kernel_spatial_dimensions()) {
      if (dim == reshape_degenerate->inserted_dimensions[0]) {
        return true;
      }
    }
  }
  return false;
}

bool LooksLikeForwardConvolution(const HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  const Shape& lhs_shape = conv->operand(0)->shape();
  const Shape& rhs_shape = conv->operand(1)->shape();
  const Shape& result_shape = conv->shape();

  int64_t lhs_batches = lhs_shape.dimensions(dnums.input_batch_dimension());
  int64_t result_batches =
      result_shape.dimensions(dnums.output_batch_dimension());
  if (lhs_batches != result_batches) {
    return false;
  }

  int64_t rhs_output_features =
      rhs_shape.dimensions(dnums.kernel_output_feature_dimension());
  int64_t result_output_features =
      result_shape.dimensions(dnums.output_feature_dimension());
  if (rhs_output_features != result_output_features) {
    return false;
  }

  for (int i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    int64_t kdim = rhs_shape.dimensions(dnums.kernel_spatial_dimensions(i));
    int64_t odim = result_shape.dimensions(dnums.output_spatial_dimensions(i));
    if (kdim > odim) {
      return false;
    }
  }

  return true;
}

ConvolutionMatch MatchBackwardInput(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward input: " << conv->name();

  if (conv->feature_group_count() > 1) {
    VLOG(1) << "Skip backward input: feature_group_count > 1";
    return std::nullopt;
  }

  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  HloInstruction* reverse_filter = conv->mutable_operand(1);
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();

  auto kernel_out_feature_dim = dnums.kernel_output_feature_dimension();
  auto kernel_out_features =
      reverse_filter->shape().dimensions(kernel_out_feature_dim);

  if (conv->feature_group_count() > 1 &&
      kernel_out_features == conv->feature_group_count()) {
    return std::nullopt;
  }

  bool is_reversed_filter =
      HloPredicateIsOp<HloOpcode::kReverse>(reverse_filter) &&
      absl::c_is_permutation(dnums.kernel_spatial_dimensions(),
                             reverse_filter->dimensions());

  bool is_reversed_conv1d_filter =
      MaybeConv1dToConv2d(conv) &&
      reverse_filter->operand(0)->opcode() == HloOpcode::kReverse;

  bool is_1x1_filter =
      absl::c_all_of(conv->window().dimensions(),
                     [](const WindowDimension& d) { return d.size() == 1; });

  if (!is_reversed_filter && !is_reversed_conv1d_filter &&
      !(window_util::HasBaseDilation(conv->window()) &&
        (reverse_filter->IsConstant() || is_1x1_filter))) {
    VLOG(1) << "Can't match to backward input convolution. Either filter is "
               "not kReverse, or it's not a base-dilated conv with a 1x1 "
               "or constant filter.";
    return std::nullopt;
  }

  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Skip backward input: stride != 1";
      return std::nullopt;
    }
    if (window_dim.window_dilation() != 1) {
      VLOG(1) << "Skip backward input: window_dilation != 1";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Skip backward input: window_reversal is set";
      return std::nullopt;
    }
  }

  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();

  const Window& old_window = conv->window();
  Window new_window = old_window;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    auto dim = new_window.mutable_dimensions(i);
    dim->set_stride(old_window.dimensions(i).base_dilation());
    dim->set_base_dilation(1);

    auto kernel_size = old_window.dimensions(i).size();
    auto backward_padding_low =
        kernel_size - 1 - old_window.dimensions(i).padding_low();
    if (backward_padding_low < 0) {
      VLOG(1) << "Skip backward input: negative padding_low";
      return std::nullopt;
    }
    dim->set_padding_low(backward_padding_low);

    auto unpadded_input_size = conv->shape().dimensions(output_spatial_dims[i]);
    auto output_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    auto padded_input_size = kernel_size + dim->stride() * (output_size - 1);
    auto total_pad_size = padded_input_size - unpadded_input_size;
    auto min_padding_high = total_pad_size - backward_padding_low;
    auto max_padding_high = min_padding_high + dim->stride() - 1;

    if (backward_padding_low >= min_padding_high &&
        backward_padding_low <= max_padding_high) {
      dim->set_padding_high(backward_padding_low);
    } else {
      if (backward_padding_low < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }

    if (dim->padding_high() < 0) {
      VLOG(1) << "Skip backward input: negative padding_high";
      return std::nullopt;
    }
  }

  auto conv_dnums = conv->convolution_dimension_numbers();
  dnums.set_kernel_input_feature_dimension(
      conv_dnums.kernel_output_feature_dimension());
  dnums.set_kernel_output_feature_dimension(
      conv_dnums.kernel_input_feature_dimension());
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    dnums.set_input_spatial_dimensions(i,
                                       conv_dnums.output_spatial_dimensions(i));
    dnums.set_output_spatial_dimensions(i,
                                        conv_dnums.input_spatial_dimensions(i));
  }
  dnums.set_input_feature_dimension(conv_dnums.output_feature_dimension());
  dnums.set_input_batch_dimension(conv_dnums.output_batch_dimension());
  dnums.set_output_feature_dimension(conv_dnums.input_feature_dimension());
  dnums.set_output_batch_dimension(conv_dnums.input_batch_dimension());

  HloInstruction* rhs = reverse_filter;
  if (HloPredicateIsOp<HloOpcode::kReverse>(rhs)) {
    rhs = rhs->mutable_operand(0);
  } else if (is_reversed_conv1d_filter) {
    auto src = rhs->mutable_operand(0)->mutable_operand(0);
    rhs = conv->parent()->AddInstruction(
        HloInstruction::CreateReshape(rhs->shape(), src));
  }

  VLOG(2) << "Matched backward input convolution: " << conv->name();
  return std::make_tuple(new_window, dnums, rhs);
}

ConvolutionMatch MatchBackwardFilter(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward filter: " << conv->name();

  if (conv->feature_group_count() > 1) {
    VLOG(1) << conv->ToString()
            << " is a forward convolution. All grouped backward filters are "
               "mapped to batch grouped convolutions in tf2xla bridge. Hence "
               "backward filter convolutions cannot have feature groups "
               "greater than 1 at this point.";
    return std::nullopt;
  }

  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  if (LooksLikeForwardConvolution(conv)) {
    VLOG(1) << "Convolution " << conv->ToString()
            << " looks like a forward convolution; skipping backward filter "
               "rewrite.";
    return std::nullopt;
  }

  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto input_feature_dim = conv_dnums.input_feature_dimension();
  auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
  auto kernel_input_feature_dim = conv_dnums.kernel_input_feature_dimension();
  auto kernel_output_feature_dim = conv_dnums.kernel_output_feature_dimension();
  auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto output_feature_dim = conv_dnums.output_feature_dimension();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();

  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window should have stride of 1";
      return std::nullopt;
    }
    if (window_dim.base_dilation() != 1) {
      VLOG(1) << "Forward convolution's window should have no base dilation";
      return std::nullopt;
    }
    if (window_dim.padding_low() < 0) {
      VLOG(1) << "Padding low should be non-negative";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return std::nullopt;
    }
  }

  int small_kernel_dimension_num = 0;
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    if (conv->operand(1)->shape().dimensions(kernel_spatial_dims[i]) <=
        conv->shape().dimensions(output_spatial_dims[i])) {
      small_kernel_dimension_num += 1;
    }
  }
  if ((kernel_spatial_dims.empty() || small_kernel_dimension_num > 1 ||
       (!MaybeConv1dToConv2d(conv) && small_kernel_dimension_num == 1)) &&
      !window_util::HasWindowDilation(conv->window())) {
    VLOG(1) << conv->ToString()
            << " is a regular forward convolution. No need to fold it to "
               "a backward filter convolution.";
    return std::nullopt;
  }

  Window backward_conv_window;
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    WindowDimension* dim = backward_conv_window.add_dimensions();
    int64_t filter_size = conv->shape().dimensions(output_spatial_dims[i]);
    dim->set_size(filter_size);
    dim->set_stride(conv->window().dimensions(i).window_dilation());
    dim->set_padding_low(conv->window().dimensions(i).padding_low());
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);

    int64_t input_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    int64_t output_size = conv->window().dimensions(i).size();

    int64_t padded_input_size = filter_size + (output_size - 1) * dim->stride();
    int64_t min_padding_high =
        padded_input_size - input_size - dim->padding_low();
    int64_t max_padding_high = min_padding_high + dim->stride() - 1;

    if (dim->padding_low() >= min_padding_high &&
        dim->padding_low() <= max_padding_high) {
      dim->set_padding_high(dim->padding_low());
    } else {
      if (dim->padding_low() < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }

    if (dim->padding_high() < 0) {
      VLOG(1) << "Fusing this pattern to backward filter convolution would "
                 "cause negative padding";
      return std::nullopt;
    }
  }

  ConvolutionDimensionNumbers backward_conv_dnums;
  backward_conv_dnums.set_input_batch_dimension(input_feature_dim);
  backward_conv_dnums.set_input_feature_dimension(input_batch_dim);
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_input_spatial_dimensions(input_spatial_dims[i]);
  }
  backward_conv_dnums.set_output_batch_dimension(kernel_input_feature_dim);
  backward_conv_dnums.set_output_feature_dimension(kernel_output_feature_dim);
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_output_spatial_dimensions(kernel_spatial_dims[i]);
  }
  backward_conv_dnums.set_kernel_input_feature_dimension(output_batch_dim);
  backward_conv_dnums.set_kernel_output_feature_dimension(output_feature_dim);
  for (int i = 0; i < output_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_kernel_spatial_dimensions(output_spatial_dims[i]);
  }

  HloInstruction* lhs = conv->mutable_operand(0);
  VLOG(2) << "Matched backward filter convolution: " << conv->name();
  return std::make_tuple(backward_conv_window, backward_conv_dnums, lhs);
}

// 从匹配到的窗口和维度编号生成配置
void config_from_matched_conv(AclnnConvolutionBackwardConfig* config,
                              const Window& window,
                              const ConvolutionDimensionNumbers& dnums,
                              const Shape& input_shape,
                              const Shape& weight_shape,
                              const Shape& output_shape,
                              int64_t groups) {
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
  config->groups = groups;
  config->cube_math_type = 0;

  // Convert ConvolutionDimensionNumbers to dim_labels string for format conversion
  config->dim_labels = ConvolutionDimensionNumbersToDimLabels(
      dnums, input_shape, weight_shape, output_shape);
  VLOG(2) << "Generated dim_labels: " << config->dim_labels;
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
bool MatchGradBiasReduce(HloInstruction* instr,
                         HloInstruction** grad_output,
                         absl::flat_hash_set<HloInstruction*>* visited) {
  if (instr->opcode() != HloOpcode::kReduce) {
    return false;
  }

  auto* reduce = Cast<HloReduceInstruction>(instr);
  if (reduce->dimensions().size() < 2) {
    return false;
  }

  if (reduce->init_values().size() != 1) return false;
  auto* init = reduce->init_values()[0];
  if (init->opcode() != HloOpcode::kConstant) return false;
  if (!ShapeUtil::IsScalar(init->shape())) return false;

  auto* reducer = reduce->called_computations()[0]->root_instruction();
  if (reducer->opcode() != HloOpcode::kAdd) return false;

  HloInstruction* operand = reduce->mutable_operand(0);
  visited->insert(operand);

  while (operand->opcode() == HloOpcode::kReshape ||
         operand->opcode() == HloOpcode::kBroadcast) {
    operand = operand->mutable_operand(0);
    visited->insert(operand);
  }

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

absl::StatusOr<bool> AclnnConvolutionBackwardRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {

    int backward_input_count = 0;
    int backward_filter_count = 0;

    std::vector<HloInstruction*> convolutions;
    for (auto* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConvolution) {
        convolutions.push_back(instr);
      }
    }

    for (auto* instr : convolutions) {
      if (instr->opcode() != HloOpcode::kConvolution) continue;

      if (auto match = MatchBackwardInput(instr)) {
        auto& [window, dnums, weight] = *match;
        HloInstruction* grad_output = instr->mutable_operand(0);

        VLOG(1) << "Converting gradInput conv: " << instr->name()
                << ", grad_output: " << grad_output->name()
                << ", weight: " << weight->name();

        Shape input_shape = instr->shape();

        AclnnConvolutionBackwardConfig config;
        config_from_matched_conv(&config, window, dnums,
                                 input_shape, weight->shape(),
                                 instr->shape(),
                                 instr->feature_group_count());
        config.output_mask = {true, false, false};
        config.transposed = false;
        std::vector<HloInstruction*> operands = {grad_output, weight};

        HloInstruction* custom_call =
            computation->AddInstruction(HloInstruction::CreateCustomCall(
                instr->shape(), operands,
                kAclnnConvolutionBackwardCallTarget));
        custom_call->set_raw_backend_config_string(config.ToString());
        custom_call->SetAndSanitizeName(
            absl::StrCat("aclnn-conv-backward-input.", backward_input_count++));

        VLOG(2) << "Created custom call: " << custom_call->name()
                << ", config: " << config.ToString();

        TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instr));
        changed = true;
        continue;
      }

      if (auto match = MatchBackwardFilter(instr)) {
        auto& [window, dnums, input] = *match;
        HloInstruction* grad_output = instr->mutable_operand(1);

        VLOG(1) << "Converting gradWeight conv: " << instr->name()
                << ", grad_output: " << grad_output->name()
                << ", input: " << input->name();

        Shape weight_shape = instr->shape();

        AclnnConvolutionBackwardConfig config;
        config_from_matched_conv(&config, window, dnums,
                                 input->shape(), weight_shape,
                                 instr->shape(),
                                 instr->feature_group_count());
        config.output_mask = {false, true, false};
        config.transposed = false;

        std::vector<HloInstruction*> operands = {grad_output, input};
        HloInstruction* custom_call =
            computation->AddInstruction(HloInstruction::CreateCustomCall(
                instr->shape(), operands,
                kAclnnConvolutionBackwardCallTarget));
        custom_call->set_raw_backend_config_string(config.ToString());
        custom_call->SetAndSanitizeName(
            absl::StrCat("aclnn-conv-backward-filter.", backward_filter_count++));

        VLOG(2) << "Created custom call: " << custom_call->name()
                << ", config: " << config.ToString();

        TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(custom_call));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instr));
        changed = true;
        continue;
      }
    }
  }

  return changed;
}

}  // namespace ascend
}  // namespace xla

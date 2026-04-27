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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace ascend {
namespace {

namespace m = match;

constexpr char kAclnnConvolutionBackwardCallTarget[] = "__aclnn$convolution_backward";

struct ConvolutionBackwardConfig {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation = {1, 1};
  bool transposed = true;  // backward uses transposed convolution
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups = 1;
  int8_t cube_math_type = 0;
  
  std::string ToString() const {
    std::string stride_str, padding_str, dilation_str, output_padding_str;
    for (int64_t s : stride) stride_str += absl::StrCat(s, ",");
    for (int64_t p : padding) padding_str += absl::StrCat(p, ",");
    for (int64_t d : dilation) dilation_str += absl::StrCat(d, ",");
    for (int64_t o : output_padding) output_padding_str += absl::StrCat(o, ",");
    
    return absl::StrCat(
        "stride=", stride_str, 
        "padding=", padding_str, 
        "dilation=", dilation_str, 
        "transposed=", transposed ? "true" : "false", 
        "output_padding=", output_padding_str, 
        "groups=", groups, 
        "cube_math_type=", static_cast<int>(cube_math_type));
  }
};

absl::Status SetName(HloModule* module, HloInstruction* instr) {
  module->SetAndUniquifyInstrName(instr, "aclnn-convolution-backward");
  return absl::OkStatus();
}

bool IsReverseOf(const HloInstruction* reverse, const HloInstruction* target) {
  if (reverse->opcode() != HloOpcode::kReverse) {
    return false;
  }
  return reverse->operand(0) == target;
}

bool IsTransposedConvolution(const HloInstruction* conv) {
  auto* conv_instr = Cast<HloConvolutionInstruction>(conv);
  const auto& dnums = conv_instr->convolution_dimension_numbers();
  
  // For transposed convolution, output feature dimension relates to input feature dimension
  // In b01f_01oi->b01f format:
  // - b: batch
  // - o: output feature
  // - i: input feature
  // - f: filter dimensions
  
  // Check if it's using "transposed" dimension numbers
  // In transposed conv, input feature (i) comes from output feature (o)
  return dnums.output_feature_dimension() == dnums.kernel_input_feature_dimension();
}

const HloInstruction* FindWeightOperand(const HloInstruction* conv, 
                                         const HloInstruction* exclude_reverse = nullptr) {
  for (int i = 0; i < conv->operand_count(); ++i) {
    const auto* operand = conv->operand(i);
    if (operand == exclude_reverse) continue;
    
    // The weight tensor typically has a different shape than the input
    // Input is usually (N, C, H, W) or similar
    // Weight is (C_out, C_in, K, K) or similar
    if (operand->shape().rank() == 4) {
      // Could be weight - check if it's a parameter or constant
      if (operand->opcode() == HloOpcode::kParameter ||
          operand->opcode() == HloOpcode::kConstant ||
          operand->opcode() == HloOpcode::kBitcast ||
          operand->opcode() == HloOpcode::kReshape) {
        return operand;
      }
    }
  }
  return nullptr;
}

const HloInstruction* FindGradOperand(const HloInstruction* conv,
                                       const HloInstruction* forward_output_grad) {
  // The grad operand should trace back to the forward's output gradient
  // This is typically the operand that has a shape matching forward output
  for (int i = 0; i < conv->operand_count(); ++i) {
    const auto* operand = conv->operand(i);
    
    // If operand traces back to the forward output gradient (through reshape etc)
    if (operand == forward_output_grad) {
      return operand;
    }
    
    // Check through common transformations
    if (operand->opcode() == HloOpcode::kReshape || 
        operand->opcode() == HloOpcode::kBitcast ||
        operand->opcode() == HloOpcode::kTranspose) {
      if (operand->operand(0) == forward_output_grad) {
        return operand;
      }
    }
  }
  return nullptr;
}

class ConvolutionBackwardInfo {
 public:
  const HloInstruction* grad_input_conv = nullptr;  // convolution computing gradInput
  const HloInstruction* grad_weight_conv = nullptr;  // convolution computing gradWeight
  
  const HloInstruction* forward_input = nullptr;  // Original forward input
  const HloInstruction* forward_weight = nullptr;  // Original forward weight
  const HloInstruction* forward_output_grad = nullptr;  // Gradient of forward output
  
  bool IsComplete() const {
    return grad_input_conv != nullptr && grad_weight_conv != nullptr &&
           forward_input != nullptr && forward_weight != nullptr;
  }
};

class AclnnConvolutionBackwardRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleConvolution(HloInstruction* instr) override {
    auto* conv = Cast<HloConvolutionInstruction>(instr);
    
    // Try to identify if this is a gradInput or gradWeight computation
    if (TryMatchGradInput(conv).ok()) {
      return absl::OkStatus();
    }
    
    if (TryMatchGradWeight(conv).ok()) {
      return absl::OkStatus();
    }
    
    return absl::OkStatus();
  }
  
  absl::Status TryMatchGradInput(const HloConvolutionInstruction* conv) {
    // gradInput pattern: convolution(gradOutput, reverse(weight))
    // The weight operand should be wrapped in reverse
    
    const HloInstruction* weight = nullptr;
    const HloInstruction* grad = nullptr;
    const HloInstruction* reverse_weight = nullptr;
    
    for (int i = 0; i < conv->operand_count(); ++i) {
      const auto* operand = conv->operand(i);
      
      // Check if this operand is reverse(something)
      if (operand->opcode() == HloOpcode::kReverse) {
        reverse_weight = operand;
        weight = operand->operand(0);
      }
    }
    
    if (!reverse_weight || !weight) {
      return absl::InternalError("Cannot find reverse(weight) pattern for gradInput");
    }
    
    // The other operand is gradOutput
    for (int i = 0; i < conv->operand_count(); ++i) {
      const auto* operand = conv->operand(i);
      if (operand != reverse_weight) {
        grad = operand;
        break;
      }
    }
    
    if (!grad) {
      return absl::InternalError("Cannot find gradOutput operand for gradInput");
    }
    
    // Store the gradInput computation info
    // We'll look for the corresponding gradWeight later
    auto key = std::make_pair(weight, grad);
    grad_input_computations_[key] = conv;
    
    VLOG(2) << "Found gradInput pattern: convolution(grad, reverse(weight))";
    VLOG(2) << "  weight: " << weight->name();
    VLOG(2) << "  grad: " << grad->name();
    
    return absl::OkStatus();
  }
  
  absl::Status TryMatchGradWeight(const HloConvolutionInstruction* conv) {
    // gradWeight pattern: convolution(forwardInput, gradOutput)
    // with transposed convolution dimension numbers
    
    if (!IsTransposedConvolution(conv)) {
      return absl::InternalError("Not a transposed convolution for gradWeight");
    }
    
    // Find the two operands
    const HloInstruction* input = nullptr;
    const HloInstruction* grad = nullptr;
    
    for (int i = 0; i < conv->operand_count(); ++i) {
      const auto* operand = conv->operand(i);
      if (operand->shape().rank() == 4) {
        // One is forward input (N, C, H, W), other is grad (same shape as output)
        if (input == nullptr) {
          input = operand;
        } else {
          grad = operand;
          break;
        }
      }
    }
    
    if (!input || !grad) {
      return absl::InternalError("Cannot find operands for gradWeight");
    }
    
    // Look for corresponding gradInput computation with same weight
    // The key is (weight, grad)
    const HloInstruction* weight = FindWeightForGradWeight(conv);
    if (!weight) {
      return absl::InternalError("Cannot find original weight for gradWeight");
    }
    
    auto key = std::make_pair(weight, grad);
    grad_weight_computations_[key] = conv;
    
    // Check if we have both gradInput and gradWeight for this convolution
    auto grad_input_it = grad_input_computations_.find(key);
    if (grad_input_it != grad_input_computations_.end()) {
      // We have both! Create the fused custom call
      TF_RETURN_IF_ERROR(CreateFusedConvolutionBackward(
          grad_input_it->second, conv, input, weight, grad));
    }
    
    VLOG(2) << "Found gradWeight pattern: convolution(input, grad)";
    VLOG(2) << "  input: " << input->name();
    VLOG(2) << "  grad: " << grad->name();
    VLOG(2) << "  weight: " << weight->name();
    
    return absl::OkStatus();
  }
  
  const HloInstruction* FindWeightForGradWeight(const HloConvolutionInstruction* conv) {
    // In gradWeight computation, one operand traces back to forward input
    // The other traces back to grad
    // We need to find the original weight parameter
    
    for (int i = 0; i < conv->operand_count(); ++i) {
      const auto* operand = conv->operand(i);
      
      // Look for parameter - this should be the weight
      if (operand->opcode() == HloOpcode::kParameter) {
        return operand;
      }
      
      // Check through transformations
      const HloInstruction* traced = TraceToParameter(operand);
      if (traced && traced->opcode() == HloOpcode::kParameter) {
        // Check if it's a weight parameter (typically has 4 dimensions)
        if (traced->shape().rank() == 4) {
          return traced;
        }
      }
    }
    return nullptr;
  }
  
  const HloInstruction* TraceToParameter(const HloInstruction* instr) {
    if (instr->opcode() == HloOpcode::kParameter) {
      return instr;
    }
    if (instr->operand_count() > 0) {
      return TraceToParameter(instr->operand(0));
    }
    return nullptr;
  }
  
  absl::Status CreateFusedConvolutionBackward(
      const HloInstruction* grad_input_conv,
      const HloInstruction* grad_weight_conv,
      const HloInstruction* forward_input,
      const HloInstruction* forward_weight,
      const HloInstruction* forward_output_grad) {
    
    // Extract configuration from either convolution (they should be similar)
    auto* conv = Cast<HloConvolutionInstruction>(grad_input_conv);
    ConvolutionConfig config = ExtractConvolutionConfig(conv);
    
    // Create tuple shape for (gradInput, gradWeight)
    Shape grad_input_shape = grad_input_conv->shape();
    Shape grad_weight_shape = grad_weight_conv->shape();
    Shape tuple_shape = ShapeUtil::MakeTupleShape({grad_input_shape, grad_weight_shape});
    
    // Create operands: input, weight, gradOutput
    std::vector<HloInstruction*> operands = {
        const_cast<HloInstruction*>(forward_input),
        const_cast<HloInstruction*>(forward_weight),
        const_cast<HloInstruction*>(forward_output_grad)
    };
    
    // Create the custom call
    HloInstruction* custom_call = grad_input_conv->AddInstruction(
        HloInstruction::CreateCustomCall(
            tuple_shape,
            operands,
            kAclnnConvolutionBackwardCallTarget));
    
    custom_call->set_raw_backend_config_string(config.ToString());
    TF_RETURN_IF_ERROR(SetName(grad_input_conv->GetModule(), custom_call));
    
    // Replace the two original convolutions with get-tuple-element of the custom call
    HloInstruction* grad_input = grad_input_conv->AddInstruction(
        HloInstruction::CreateGetTupleElement(grad_input_shape, custom_call, 0));
    
    HloInstruction* grad_weight = grad_weight_conv->AddInstruction(
        HloInstruction::CreateGetTupleElement(grad_weight_shape, custom_call, 1));
    
    TF_RETURN_IF_ERROR(ReplaceInstruction(grad_input_conv, grad_input));
    TF_RETURN_IF_ERROR(ReplaceInstruction(grad_weight_conv, grad_weight));
    
    VLOG(2) << "Created fused convolution backward custom call";
    
    return absl::OkStatus();
  }
  
  ConvolutionConfig ExtractConvolutionConfig(const HloConvolutionInstruction* conv) {
    ConvolutionConfig config;
    
    const Window& window = conv->window();
    
    // Extract stride for spatial dimensions
    for (size_t i = 1; i < window.dimensions().size(); ++i) {
      config.stride.push_back(window.dimensions(i).stride());
    }
    
    // Extract padding (only spatial dimensions, low and high)
    for (size_t i = 1; i < window.dimensions().size(); ++i) {
      const auto& dim = window.dimensions(i);
      config.padding.push_back(dim.padding_low());
      config.padding.push_back(dim.padding_high());
    }
    
    // Dilation (default to 1 for all spatial dimensions)
    config.dilation = std::vector<int64_t>(config.stride.size(), 1);
    
    // Transposed is true for backward
    config.transposed = true;
    
    // Output padding (default to 0 for all spatial dimensions)
    config.output_padding = std::vector<int64_t>(config.stride.size(), 0);
    
    // Groups (default to 1)
    config.groups = 1;
    
    return config;
  }

 private:
  std::unordered_map<std::pair<const HloInstruction*, const HloInstruction*>, 
                     const HloInstruction*, 
                     PairHash> grad_input_computations_;
  std::unordered_map<std::pair<const HloInstruction*, const HloInstruction*>, 
                     const HloInstruction*, 
                     PairHash> grad_weight_computations_;
};

struct PairHash {
  size_t operator()(const std::pair<const HloInstruction*, const HloInstruction*>& p) const {
    return std::hash<const void*>()(p.first) ^ std::hash<const void*>()(p.second);
  }
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  AclnnConvolutionBackwardRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

absl::StatusOr<bool> AclnnConvolutionBackwardRewriter::RunImpl(
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
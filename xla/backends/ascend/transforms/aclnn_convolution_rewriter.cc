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

namespace xla {
namespace ascend {
namespace {

namespace m = match;

constexpr char kAclnnConvolutionCallTarget[] = "__aclnn$convolution";

struct ConvolutionConfig {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation = {1, 1};
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups = 1;
  int8_t cube_math_type = 0;  // 0: KEEP_DTYPE
  bool has_bias = false;

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
        "cube_math_type=", static_cast<int>(cube_math_type), 
        "has_bias=", has_bias ? "true" : "false");
  }
};

absl::Status SetName(HloModule* module, HloInstruction* conv) {
  module->SetAndUniquifyInstrName(conv, "aclnn-convolution");
  return absl::OkStatus();
}

template <typename Pattern>
auto OptionalReshape(HloInstruction** optional_reshape, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Reshape(optional_reshape, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalBroadcast(HloInstruction** optional_broadcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Broadcast(optional_broadcast, pattern),
                                  std::move(pattern));
}

bool IsBiasCompatibleWithOutput(const HloInstruction* bias, const HloInstruction* conv) {
  const Shape& bias_shape = bias->shape();
  const Shape& conv_shape = conv->shape();
  
  if (bias_shape.rank() == 1) {
    // Bias shape should match the output channel dimension
    int64_t output_channel_dim = 3;  // NHWC format
    if (conv_shape.rank() == 4) {
      return bias_shape.dimensions(0) == conv_shape.dimensions(output_channel_dim);
    }
  }
  return false;
}

ConvolutionConfig ExtractConvolutionConfig(const HloConvolutionInstruction* conv) {
  ConvolutionConfig config;
  
  // Extract stride for spatial dimensions
  for (size_t i = 1; i < conv->window().dimensions().size(); ++i) {
    config.stride.push_back(conv->window().dimensions(i).stride());
  }
  
  // Extract padding (only spatial dimensions, low and high)
  for (size_t i = 1; i < conv->window().dimensions().size(); ++i) {
    const auto& dim = conv->window().dimensions(i);
    config.padding.push_back(dim.padding_low());
    config.padding.push_back(dim.padding_high());
  }
  
  // Dilation (default to 1 for all spatial dimensions)
  config.dilation = std::vector<int64_t>(config.stride.size(), 1);
  
  // Transposed (default to false)
  config.transposed = false;
  
  // Output padding (default to 0 for all spatial dimensions)
  config.output_padding = std::vector<int64_t>(config.stride.size(), 0);
  
  // Groups (default to 1)
  config.groups = 1;
  
  return config;
}

class AclnnConvolutionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleConvolution(HloInstruction* instr) override {
    auto* conv = Cast<HloConvolutionInstruction>(instr);
    
    // Create custom call for convolution without bias
    std::vector<HloInstruction*> operands = {
        conv->mutable_operand(0),  // input
        conv->mutable_operand(1)   // weight
    };
    
    ConvolutionConfig config = ExtractConvolutionConfig(conv);
    config.has_bias = false;
    
    HloInstruction* conv_call = instr->AddInstruction(
        HloInstruction::CreateCustomCall(
            instr->shape(),
            operands,
            kAclnnConvolutionCallTarget));
    
    conv_call->set_raw_backend_config_string(config.ToString());
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), conv_call));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, conv_call));
    
    return absl::OkStatus();
  }
  
  absl::Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *bias = nullptr, *conv = nullptr;
    HloInstruction* optional_reshape = nullptr;
    HloInstruction* optional_broadcast = nullptr;
    
    // Match pattern: add(conv, broadcast(bias))
    if (Match(instr,
              m::AddAnyOrder(
                  m::Convolution(&conv).WithOneUser(),
                  OptionalReshape(
                      &optional_reshape,
                      OptionalBroadcast(
                          &optional_broadcast,
                          m::Parameter(&bias)
                      )
                  )
              ))) {
      if (!IsBiasCompatibleWithOutput(bias, conv)) {
        return absl::OkStatus();
      }
      
      // Create new custom call with bias
      std::vector<HloInstruction*> operands = {
          conv->mutable_operand(0),  // input
          conv->mutable_operand(1),  // weight
          bias                       // bias
      };
      
      ConvolutionConfig config = ExtractConvolutionConfig(Cast<HloConvolutionInstruction>(conv));
      config.has_bias = true;
      
      HloInstruction* fused_conv = instr->AddInstruction(
          HloInstruction::CreateCustomCall(
              instr->shape(),
              operands,
              kAclnnConvolutionCallTarget));
      
      fused_conv->set_raw_backend_config_string(config.ToString());
      TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_conv));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, fused_conv));
      
      return absl::OkStatus();
    }
    
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  AclnnConvolutionRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
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
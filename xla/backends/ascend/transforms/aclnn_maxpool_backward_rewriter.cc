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

#include "xla/backends/ascend/transforms/aclnn_maxpool_backward_rewriter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
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

namespace xla {
namespace ascend {
namespace {

namespace m = match;

constexpr char kAclnnMaxPool2dWithIndicesBackwardCallTarget[] = "__aclnn$max_pool2d_with_indices_backward";

struct MaxPoolConfig {
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation = {1, 1};
  bool ceil_mode = false;

  std::string ToString() const {
    return absl::StrCat(
        "kernel_size=", kernel_size[0], ",", kernel_size[1],
        ",stride=", stride[0], ",", stride[1],
        ",padding=", padding[0], ",", padding[1],
        ",dilation=", dilation[0], ",", dilation[1],
        ",ceil_mode=", ceil_mode ? "true" : "false");
  }
};

bool IsMaxPoolSelectComputation(const HloComputation* select) {
  if (select->instruction_count() != 3) {
    return false;
  }
  auto instructions = select->instructions();
  HloInstruction* root = select->root_instruction();
  if (root->opcode() != HloOpcode::kCompare) {
    return false;
  }
  auto compare = Cast<HloCompareInstruction>(root);
  if (compare->direction() != Comparison::Direction::kGe) {
    return false;
  }
  return true;
}

bool IsMaxPoolScatterComputation(const HloComputation* scatter) {
  if (scatter->instruction_count() != 3) {
    return false;
  }
  auto instructions = scatter->instructions();
  HloInstruction* root = scatter->root_instruction();
  if (root->opcode() != HloOpcode::kAdd) {
    return false;
  }
  return true;
}

MaxPoolConfig ExtractMaxPoolConfig(const HloSelectAndScatterInstruction* sas) {
  MaxPoolConfig config;
  const Window& window = sas->window();
  
  for (const WindowDimension& dim : window.dimensions()) {
    if (dim.size() > 1) {
      config.kernel_size.push_back(dim.size());
      config.stride.push_back(dim.stride());
      config.padding.push_back(dim.padding_low());
    }
  }
  
  if (config.kernel_size.size() != 2) {
    config.kernel_size = {2, 2};
    config.stride = {2, 2};
    config.padding = {0, 0};
  }
  
  return config;
}

class AclnnMaxPoolBackwardRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleSelectAndScatter(HloInstruction* instr) override {
    auto* sas = Cast<HloSelectAndScatterInstruction>(instr);
    
    if (!IsMaxPoolSelectComputation(sas->select())) {
      return absl::OkStatus();
    }
    
    if (!IsMaxPoolScatterComputation(sas->scatter())) {
      return absl::OkStatus();
    }
    
    HloInstruction* operand = sas->mutable_operand(0);
    HloInstruction* source = sas->mutable_operand(1);
    
    MaxPoolConfig config = ExtractMaxPoolConfig(sas);
    
    std::vector<HloInstruction*> operands = {source, operand};
    
    HloInstruction* custom_call = instr->AddInstruction(
        HloInstruction::CreateCustomCall(
            instr->shape(),
            operands,
            kAclnnMaxPool2dWithIndicesBackwardCallTarget));
    
    custom_call->set_raw_backend_config_string(config.ToString());
    
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, custom_call));
    
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  AclnnMaxPoolBackwardRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

absl::StatusOr<bool> AclnnMaxPoolBackwardRewriter::RunImpl(
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
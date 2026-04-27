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

#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONVOLUTION_BACKWARD_REWRITER_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONVOLUTION_BACKWARD_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace ascend {

// A pass that rewrites convolution backward operations to use aclnnConvolutionBackward.
//
// This pass identifies transposed convolution patterns used in backward pass computation:
// 1. gradInput: convolution with reverse(weight) -> input gradient
// 2. gradWeight: transposed convolution without reverse -> weight gradient
//
// The pass creates custom calls that can invoke aclnnConvolutionBackward to compute
// both gradients in a single kernel call.
class AclnnConvolutionBackwardRewriter : public HloModulePass {
 public:
  AclnnConvolutionBackwardRewriter() = default;
  absl::string_view name() const override { return "aclnn-convolution-backward-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONVOLUTION_BACKWARD_REWRITER_H_
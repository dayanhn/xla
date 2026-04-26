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

#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_GEMM_REWRITER_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_GEMM_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace ascend {

// Ascend ACLNN GEMM rewriter pass that rewrites HLO dot operations into
// custom calls that can be executed by the ACLNN library.
//
// The ACLNN GEMM interface supports:
//   out = alpha * (A @ B) + beta * C
//
// This pass pattern-matches the following patterns and rewrites them into
// a single custom call:
//
// 1. Simple GEMM:
//    dot(A, B) -> custom_call("__aclnn$gemm", A, B)
//
// 2. GEMM with bias (vector):
//    add(dot(A, B), broadcast(bias)) -> custom_call("__aclnn$gemm", A, B, bias)
//
// 3. GEMM with scaled output:
//    multiply(dot(A, B), alpha) -> custom_call("__aclnn$gemm", A, B, alpha=alpha)
//
// 4. GEMM with broadcast bias (complex reshape/broadcast chain):
//    add(dot(A, B), complex_broadcast_chain(bias)) -> custom_call("__aclnn$gemm", A, B, bias)
//
class AclnnGemmRewriter : public HloModulePass {
 public:
  explicit AclnnGemmRewriter(se::GpuComputeCapability gpu_version);
  absl::string_view name() const override { return "aclnn-gemm-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_GEMM_REWRITER_H_

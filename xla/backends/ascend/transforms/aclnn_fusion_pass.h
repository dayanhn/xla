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

#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_FUSION_PASS_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_FUSION_PASS_H_

#include "absl/status/status.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace ascend {

// Run ACLNN fusion pass pipeline, which includes ACLNN GEMM rewriter.
absl::Status RunAclnnFusionPass(HloModule* hlo_module, const xla::gpu::GpuTargetConfig& gpu_target_config);

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_FUSION_PASS_H_

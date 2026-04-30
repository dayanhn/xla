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

#include "xla/backends/ascend/transforms/aclnn_fusion_pass.h"

#include "xla/backends/ascend/transforms/aclnn_gemm_rewriter.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/ascend/transforms/aclnn_convolution_rewriter.h"
#include "xla/backends/ascend/transforms/aclnn_convolution_backward_rewriter.h"

namespace xla {
namespace ascend {

absl::Status RunAclnnFusionPass(HloModule* hlo_module, const xla::gpu::GpuTargetConfig& gpu_target_config) {
  se::GpuComputeCapability gpu_version = gpu_target_config.device_description.gpu_compute_capability();
  HloPassPipeline pipeline("aclnn-fusion");

  pipeline.AddPass<AclnnGemmRewriter>(gpu_version);
  pipeline.AddPass<AclnnConvolutionBackwardRewriter>();
  pipeline.AddPass<AclnnConvolutionRewriter>(gpu_version);
  return pipeline.Run(hlo_module, {HloInstruction::kMainExecutionThread}).status();
}

}  // namespace ascend
}  // namespace xla

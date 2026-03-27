/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/ascend/autotuner/ascend_dnn.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace ascend {

AscendDnnBackend::AscendDnnBackend(stream_executor::StreamExecutor* stream_exec, 
                                   const DebugOptions* debug_options, 
                                   gpu::GpuCompiler* compiler, 
                                   const gpu::GpuTargetConfig* target_config)
    : CodegenBackend(stream_exec, debug_options, compiler, target_config) {
}

bool AscendDnnBackend::IsSupported(const HloInstruction& instr) {
  // TODO: Implement support check for Ascend DNN operations
  return false;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> AscendDnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  // TODO: Implement default config retrieval
  return absl::UnimplementedError("AscendDnnBackend::GetDefaultConfig not implemented");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> AscendDnnBackend::GetSupportedConfigs(
    const HloInstruction& instr) {
  // TODO: Implement supported configs retrieval
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::Status AscendDnnBackend::ApplyConfig(HloInstruction& instr, 
                                         const BackendConfig& config) {
  // TODO: Implement config application
  return absl::UnimplementedError("AscendDnnBackend::ApplyConfig not implemented");
}

}  // namespace ascend
}  // namespace xla

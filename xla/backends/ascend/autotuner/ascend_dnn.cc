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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace ascend {

AscendDnnBackend::AscendDnnBackend(stream_executor::StreamExecutor* stream_executor,
                                   const DebugOptions* debug_options,
                                   Compiler* compiler,
                                   const Compiler::GpuTargetConfig* target_config)
    : stream_executor_(stream_executor),
      debug_options_(debug_options),
      compiler_(compiler),
      target_config_(target_config) {
}

absl::string_view AscendDnnBackend::name() const {
  return "AscendDnnBackend";
}

autotuner::Backend AscendDnnBackend::backend() const {
  return autotuner::Backend::ASCEND_DNN;
}

bool AscendDnnBackend::CanProduceWrongResults() const {
  return false;
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
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }
  auto config = GetDefaultConfig(instr);
  if (config.ok()) {
    configs.push_back(std::move(config.value()));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<Executable>> AscendDnnBackend::Compile(
    const HloInstruction& instr, const BackendConfig& config) {
  // TODO: Implement compilation logic for Ascend DNN
  return absl::UnimplementedError("AscendDnnBackend::Compile not implemented");
}

absl::Status AscendDnnBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  // TODO: Implement config application
  return absl::UnimplementedError("AscendDnnBackend::ApplyConfig not implemented");
}

}  // namespace ascend
}  // namespace xla

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

#ifndef XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_
#define XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/service/compiler.h"

namespace xla {
namespace ascend {

class AscendDnnBackend : public CodegenBackend {
 public:
  AscendDnnBackend(stream_executor::StreamExecutor* stream_exec, 
                   const DebugOptions* debug_options, 
                   gpu::GpuCompiler* compiler, 
                   const gpu::GpuTargetConfig* target_config);

  bool IsSupported(const HloInstruction& instr) override;
  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> GetSupportedConfigs(
      const HloInstruction& instr) override;
  absl::Status ApplyConfig(HloInstruction& instr, 
                          const BackendConfig& config) override;
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_

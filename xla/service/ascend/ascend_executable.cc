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

#include "xla/service/ascend/ascend_executable.h"

namespace xla {

AscendExecutable::AscendExecutable(
    std::unique_ptr<HloModule> hlo_module,
    se::StreamExecutor* stream_executor)
    : GpuExecutable(std::move(hlo_module), stream_executor) {
}

AscendExecutable::~AscendExecutable() {
}

absl::StatusOr<ExecutionOutput> AscendExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  // TODO: Implement Ascend-specific execution logic
  return GpuExecutable::ExecuteAsyncOnStream(
      run_options, std::move(arguments), hlo_execution_profile);
}

}  // namespace xla
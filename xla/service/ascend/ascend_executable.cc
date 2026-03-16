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
    : Executable(std::move(hlo_module)),
      stream_executor_(stream_executor) {
}

AscendExecutable::~AscendExecutable() {
}

absl::StatusOr<std::unique_ptr<AscendExecutable>> AscendExecutable::Create(
    std::unique_ptr<HloModule> hlo_module,
    se::StreamExecutor* stream_executor) {
  return std::make_unique<AscendExecutable>(
      std::move(hlo_module), stream_executor);
}

absl::StatusOr<ExecutionOutput> AscendExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  // TODO: Implement Ascend-specific execution
  return absl::UnimplementedError("ExecuteAsyncOnStream not implemented for Ascend");
}

int64_t AscendExecutable::SizeOfGeneratedCodeInBytes() const {
  // TODO: Implement Ascend-specific code size calculation
  return 0;
}

absl::Span<const BufferAllocation* absl_nonnull const> AscendExecutable::GetAllocations()
    const {
  // TODO: Implement Ascend-specific allocation retrieval
  static std::vector<const BufferAllocation*> empty;
  return empty;
}

}  // namespace xla

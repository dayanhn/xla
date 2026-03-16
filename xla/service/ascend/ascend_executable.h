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

#ifndef XLA_SERVICE_ASCEND_ASCEND_EXECUTABLE_H_
#define XLA_SERVICE_ASCEND_ASCEND_EXECUTABLE_H_

#include "xla/service/executable.h"

namespace xla {

// Executable for Ascend devices.
class AscendExecutable : public Executable {
 public:
  // Creates an AscendExecutable.
  static absl::StatusOr<std::unique_ptr<AscendExecutable>> Create(
      std::unique_ptr<HloModule> hlo_module,
      se::StreamExecutor* stream_executor);

  ~AscendExecutable() override;

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  absl::Span<const BufferAllocation* absl_nonnull const> GetAllocations()
      const override;

 public:
  AscendExecutable(
      std::unique_ptr<HloModule> hlo_module,
      se::StreamExecutor* stream_executor);

  private:
  se::StreamExecutor* stream_executor_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASCEND_ASCEND_EXECUTABLE_H_
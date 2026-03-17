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

#include "xla/util.h"

namespace xla {

AscendExecutable::~AscendExecutable() {
}

absl::StatusOr<std::unique_ptr<AscendExecutable>> AscendExecutable::Create(
    gpu::GpuExecutable::Params params) {
  // Create a GpuExecutable first
  auto gpu_executable_or = gpu::GpuExecutable::Create(std::move(params));
  if (!gpu_executable_or.ok()) {
    return gpu_executable_or.status();
  }
  auto gpu_executable = std::move(gpu_executable_or.value());
  
  // Convert it to AscendExecutable
  return std::unique_ptr<AscendExecutable>(
      static_cast<AscendExecutable*>(gpu_executable.release()));
}

}  // namespace xla

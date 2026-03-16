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

#include "xla/service/ascend/ascend_compiler.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/ascend/ascend_platform_id.h"

namespace xla {

static bool InitModule() {
  Compiler::RegisterCompilerFactory(stream_executor::ascend::kAscendPlatformId,
                         []() -> absl::StatusOr<std::unique_ptr<Compiler>> {
                           return std::make_unique<AscendCompiler>();
                         });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace xla
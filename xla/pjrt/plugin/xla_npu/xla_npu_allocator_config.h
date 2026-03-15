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

#ifndef XLA_PJRT_PLUGIN_XLA_NPU_XLA_NPU_ALLOCATOR_CONFIG_H_
#define XLA_PJRT_PLUGIN_XLA_NPU_XLA_NPU_ALLOCATOR_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "xla/tsl/framework/allocator.h"

namespace xla {

struct NpuAllocatorConfig {
  enum class Kind {
    kDefault,
    kBFC,
    kPlatform,
  };

  Kind kind = Kind::kDefault;
  double memory_fraction = 0.9;
  bool preallocate = false;
  std::optional<int64_t> npu_system_memory_size;
  std::vector<tsl::SubAllocator::Visitor> sub_allocator_alloc_visitors;
  std::vector<tsl::SubAllocator::Visitor> sub_allocator_free_visitors;
  size_t collective_memory_size = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_NPU_XLA_NPU_ALLOCATOR_CONFIG_H_
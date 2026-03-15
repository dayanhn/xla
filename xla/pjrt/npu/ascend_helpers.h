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

#ifndef XLA_PJRT_NPU_ASCEND_HELPERS_H_#define XLA_PJRT_NPU_ASCEND_HELPERS_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <map>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

// Helper function to get Ascend platform
absl::StatusOr<se::Platform*> GetAscendPlatform();

// Helper function to get Ascend XLA client
absl::StatusOr<LocalClient*> GetAscendXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices);



// Helper function to enable peer access between Ascend devices
void EnableAscendPeerAccess(absl::Span<se::StreamExecutor* const> executors);

// Helper function to get Ascend host allocator
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> GetAscendHostAllocator(
    se::StreamExecutor* executor);

}  // namespace xla

#endif  // XLA_PJRT_NPU_ASCEND_HELPERS_H_
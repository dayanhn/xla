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

#include "xla/pjrt/npu/ascend_helpers.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <map>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/integrations/stream_executor_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"

namespace xla {

// Helper function to get Ascend platform
absl::StatusOr<se::Platform*> GetAscendPlatform() {
  return PlatformUtil::GetPlatform("ASCEND");
}

// Helper function to get Ascend XLA client
absl::StatusOr<LocalClient*> GetAscendXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices) {
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      GetAscendPlatform());
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible Ascend devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  options.set_allowed_devices(allowed_devices);
  return ClientLibrary::GetOrCreateLocalClient(options);
}


// Helper function to enable peer access between Ascend devices
void EnableAscendPeerAccess(absl::Span<se::StreamExecutor* const> executors) {
  for (int i = 0; i < executors.size(); ++i) {
    for (int j = 0; j < executors.size(); ++j) {
      if (i == j) {
        continue;
      }
      se::StreamExecutor* from = executors[i];
      se::StreamExecutor* to = executors[j];
      if (from->CanEnablePeerAccessTo(to)) {
        absl::Status status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          LOG(WARNING) << "Unable to enable peer access between Ascend devices " << i
                       << " and " << j << "; status: " << status;
        } else {
          VLOG(2) << "Enabled peer access from Ascend device " << i << " to device " << j;
        }
      }
    }
  }
}



// Helper function to get Ascend host allocator
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> GetAscendHostAllocator(
    se::StreamExecutor* executor) {
  TF_ASSIGN_OR_RETURN(
      auto host_memory_allocator,
      executor->CreateMemoryAllocator(stream_executor::MemorySpace::kHost));
  std::unique_ptr<tsl::SubAllocator> sub_allocator(
      new se::StreamExecutorAllocator(std::move(host_memory_allocator),
                                      stream_executor::MemorySpace::kHost,
                                      /*index=*/0,
                                      /*alloc_visitors=*/{},
                                      /*free_visitors=*/{}));
  bool xla_pjrt_ascend_host_memory_preallocate;
  TF_RETURN_IF_ERROR(
      tsl::ReadBoolFromEnvVar("XLA_PJRT_ASCEND_HOST_MEMORY_PREALLOCATE", false,
                              &xla_pjrt_ascend_host_memory_preallocate));

  const int64_t default_xla_pjrt_ascend_host_memory_limit_gb = 64;

  int64_t xla_pjrt_ascend_host_memory_limit_gb;
  TF_RETURN_IF_ERROR(
      tsl::ReadInt64FromEnvVar("XLA_PJRT_ASCEND_HOST_MEMORY_LIMIT_GB",
                               default_xla_pjrt_ascend_host_memory_limit_gb,
                               &xla_pjrt_ascend_host_memory_limit_gb));

  const int64_t kAscendHostMemoryLimitBytes = 
      xla_pjrt_ascend_host_memory_limit_gb * (1LL << 30);

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = !xla_pjrt_ascend_host_memory_preallocate;
  return std::make_unique<tsl::BFCAllocator>(std::move(sub_allocator),
                                             kAscendHostMemoryLimitBytes,
                                             /*name=*/"xla_ascend_host_bfc", opts);
}

}  // namespace xla
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

// Builds a BFCAllocator for all local Ascend devices.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction, bool preallocate,
    std::optional<int64_t> npu_system_memory_size,
    const std::vector<tsl::SubAllocator::Visitor>& sub_allocator_alloc_visitors,
    const std::vector<tsl::SubAllocator::Visitor>&
        sub_allocator_free_visitors) {
  int device_ordinal = executor->device_ordinal();
  std::unique_ptr<tsl::SubAllocator> sub_allocator = 
      std::make_unique<se::DeviceMemAllocator>(
          executor, tsl::PlatformDeviceId(device_ordinal),
          sub_allocator_alloc_visitors, sub_allocator_free_visitors);

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }

  size_t allocator_memory = total_memory * memory_fraction;
  // If npu_system_memory_size is set, use it instead of default value.
  if (npu_system_memory_size.has_value()) {
    allocator_memory = npu_system_memory_size.value();
  }

  if (preallocate) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal << " for BFCAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal << " for BFCAllocator.";
  }

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = !preallocate;
  return std::make_unique<tsl::BFCAllocator>(
      std::move(sub_allocator), allocator_memory,
      absl::StrCat("ASCEND_", device_ordinal, "_bfc"), opts);
}

// Builds a BFCAllocator for all local Ascend devices that uses collective memory.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateCollectiveBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction,
    size_t collective_memory_size) {
  int device_ordinal = executor->device_ordinal();
  std::unique_ptr<tsl::SubAllocator> sub_allocator = 
      std::make_unique<se::DeviceMemAllocator>(
          executor, tsl::PlatformDeviceId(device_ordinal));

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }
  bool preallocate = collective_memory_size != 0;
  size_t allocator_memory = 
      preallocate ? collective_memory_size : total_memory * memory_fraction;

  if (preallocate) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CollectiveBFCAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CollectiveBFCAllocator.";
  }

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = !preallocate;
  return std::make_unique<tsl::BFCAllocator>(
      std::move(sub_allocator), allocator_memory,
      absl::StrCat("ASCEND_collectivememory_", device_ordinal, "_bfc"), opts);
}



// Returns a Ascend pinned host memory allocator to use when staging host->Ascend
// transfers. We use a fixed pool of pinned memory.
//
// The pool size is controlled by XLA_PJRT_ASCEND_HOST_MEMORY_LIMIT_GB environment
// variable, which defaults to 64GB.
//
// If XLA_PJRT_ASCEND_HOST_MEMORY_PREALLOCATE is set to true, the pool will be
// preallocated, and the preallocated size is controlled by
// XLA_PJRT_ASCEND_HOST_MEMORY_LIMIT_GB environment variable, which defaults to
// 16GB in this case.
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

  const int64_t default_xla_pjrt_ascend_host_memory_limit_gb = 
      xla_pjrt_ascend_host_memory_preallocate ? 16 : 64;

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
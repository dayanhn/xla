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

#include "xla/backends/npu/collectives/hccl_collectives.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/hccl/inc/hccl/hccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/npu/collectives/hccl_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

namespace xla::npu {

// Convert HcclResult to absl::Status
absl::Status HcclStatusToAbslStatus(HcclResult result, const std::string& message) {
  if (result == HCCL_SUCCESS) {
    return absl::OkStatus();
  }
  
  const char* error_str = HcclGetErrorString(result);
  return absl::InternalError(
      absl::StrCat(message, ": HCCL error: ", 
                   error_str ? error_str : "unknown error"));
}

//===----------------------------------------------------------------------===//
// HcclIdStore
//===----------------------------------------------------------------------===//

namespace {
// HcclIdStore generates clique unique ids for NPU cliques using HCCL APIs.
// HCCL unique id is a mechanism that used by HCCL to find all processes
// participating in the collective clique.
class HcclIdStore {
 public:
  HcclIdStore(ProcessId process_id,
              absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
              std::shared_ptr<KeyValueStoreInterface> kv_store)
      : process_id_(process_id),
        device_to_process_(std::move(device_to_process)),
        kv_store_(std::move(kv_store)) {}

  absl::StatusOr<CliqueIds> GetCliqueIds(const CliqueKey& key,
                                         HcclCollectives& hccl_collectives) {
    auto* gpu_key = tsl::down_cast<const gpu::GpuCliqueKey*>(&key);
    if (gpu_key == nullptr) {
      return InvalidArgument("Expected GPU clique key");
    }

    // The caller must ensure that threads calling this method concurrently have
    // unique keys, otherwise the global key-value store may hold the wrong
    // value.
    {
      absl::MutexLock lock(mu_);
      auto it = cache_.find(*gpu_key);
      if (it != cache_.end()) {
        return it->second;
      }
    }

    // Check how many roots are needed to initialize the GpuClique.
    static const int64_t hccl_init_rank_per_root_ratio =
        xla::GetDebugOptionsFromFlags()
            .xla_gpu_nccl_init_max_rank_per_root_ratio();
    int64_t nranks = key.num_devices();
    int64_t nroots = hccl_init_rank_per_root_ratio != 0
                         ? CeilOfRatio(nranks, hccl_init_rank_per_root_ratio)
                         : 1;

    // Create a KV store key for the given root process and captured clique key.
    auto kv_key = [&](ProcessId root_process) {
      return absl::StrFormat("root_process: %v; clique: %v", root_process, key);
    };

    // Global devices that are responsible for generating clique ids.
    std::vector<GlobalDeviceId> root_devices = gpu_key->GetRootDevices(nroots);

    // Processes that are responsible for generating clique ids. Note that if
    // multiple root devices belong to the same process id, we will generate
    // just one clique id for them. We keep processes in a sorted container to
    // guarantee that all ranks will generate identical clique ids.
    absl::btree_set<ProcessId> root_processes;
    for (GlobalDeviceId root : root_devices) {
      root_processes.insert(device_to_process_.at(root));
    }

    VLOG(4) << absl::StreamFormat(
        "Get HCCL clique ids: process=%v; root_devices=%d:[%s]; "
        "root_processes=%d:[%s]; clique=%v",
        process_id_, root_devices.size(), HumanReadableDevices(root_devices),
        root_processes.size(),
        HumanReadableProcesses(std::vector<ProcessId>(root_processes.begin(),
                                                      root_processes.end())),
        key);

    // If we are one of the root processes, generate the key and exchange it
    // with other ranks by putting into KV store.
    if (root_processes.contains(process_id_)) {
      absl::Time set_clique_id_start = absl::Now();
      TF_ASSIGN_OR_RETURN(CliqueId clique_id,
                          hccl_collectives.CreateUniqueCliqueId());
      TF_RETURN_IF_ERROR(
          kv_store_->Set(kv_key(process_id_), clique_id.ToString()));
      VLOG(5) << absl::StreamFormat(
          "Set HCCL clique id process=%v in %s", process_id_,
          absl::FormatDuration(absl::Now() - set_clique_id_start));
    }

    // Collect generated clique ids for all root processes. We will read back
    // the key that we just generated, it's a small performance vs code
    // readbility tradeoff.
    absl::Time get_clique_ids_start = absl::Now();
    CliqueIds clique_ids;
    for (ProcessId root : root_processes) {
      TF_ASSIGN_OR_RETURN(std::string id_str,
                          kv_store_->Get(kv_key(root), absl::Minutes(10)));
      clique_ids.Add(CliqueId(id_str));
    }

    VLOG(5) << absl::StreamFormat(
        "Got HCCL clique ids in %s: root_devices=%d:[%s]; "
        "root_processes=%d:[%s]; clique=%v",
        absl::FormatDuration(absl::Now() - get_clique_ids_start),
        root_devices.size(), HumanReadableDevices(root_devices),
        root_processes.size(),
        HumanReadableProcesses(std::vector<ProcessId>(root_processes.begin(),
                                                      root_processes.end())),
        key);

    absl::MutexLock lock(mu_);
    auto result = cache_.emplace(*gpu_key, std::move(clique_ids));
    TF_RET_CHECK(result.second) << "Clique IDs already in cache";
    return result.first->second;
  }

 private:
  ProcessId process_id_;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process_;
  std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::GpuCliqueKey, CliqueIds> cache_ ABSL_GUARDED_BY(mu_);
};
}  // namespace

//===----------------------------------------------------------------------===//
// HcclCollectives
//===----------------------------------------------------------------------===//

static auto DeviceOrdinal(const Collectives::DeviceRank& rank) {
  auto* device = tsl::down_cast<const gpu::GpuCollectives::Device*>(rank.device);
  return device->stream_executor()->device_ordinal();
}

static auto DeviceOrdinalsToString(
    absl::Span<const Collectives::DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    auto* device = tsl::down_cast<const gpu::GpuCollectives::Device*>(rank.device);
    absl::StrAppend(str, device->stream_executor()->device_ordinal());
  });
}

static auto DeviceRanksToString(
    absl::Span<const Collectives::DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    absl::StrAppend(str, rank.rank.value());
  });
}

static HcclComm Cast(const Communicator* comm) {
  auto* hccl_communicator = tsl::down_cast<const HcclCommunicator*>(comm);
  CHECK(hccl_communicator != nullptr) << "Unsupported XLA communicator";
  return hccl_communicator->comm();
}

absl::StatusOr<CliqueId> HcclCollectives::CreateUniqueCliqueId() const {
  HcclRootInfo root_info;
  HcclResult result = HcclGetRootInfo(&root_info);
  if (result != HCCL_SUCCESS) {
    return HcclStatusToAbslStatus(result, "HcclGetRootInfo failed");
  }
  
  // Convert HcclRootInfo to bytes for CliqueId
  // HcclRootInfo is typically a struct containing unique identifier data
  const char* data = reinterpret_cast<const char*>(&root_info);
  return CliqueId(absl::string_view(data, sizeof(HcclRootInfo)));
}

size_t HcclCollectives::SymmetricMemoryAlignment() const {
  // TODO: Query memory alignment from Ascend NPU
  // For now, use a reasonable default alignment
  return 4096;
}

static absl::StatusOr<HcclRootInfo> AsHcclRootInfo(const CliqueId& clique_id) {
  if (clique_id.size() != sizeof(HcclRootInfo)) {
    return Internal(
        "CliqueId size is not equal to HCCL_ROOT_INFO_SIZE: %d vs %zu",
        clique_id.size(), sizeof(HcclRootInfo));
  }
  HcclRootInfo root_info;
  absl::c_copy(clique_id.data(), reinterpret_cast<char*>(&root_info));
  return root_info;
}

// Collect stream executors from all Device ranks. Returns an error if the
// device is not a `GpuCollectives` device.
static absl::StatusOr<std::vector<se::StreamExecutor*>> GetStreamExecutors(
    absl::Span<const HcclCollectives::DeviceRank> ranks) {
  std::vector<se::StreamExecutor*> stream_executors(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    auto* device = tsl::down_cast<gpu::GpuCollectives::Device*>(ranks[i].device);
    TF_RET_CHECK(device) << "Device must be GpuCollectives::Device";
    stream_executors[i] = device->stream_executor();
  }
  return stream_executors;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
HcclCollectives::CreateCommunicatorsWithCancel(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config,
    std::shared_ptr<xla::gpu::CancellationToken> cancel) {
  // Validate clique ids. With the HCCL backend, we rely on the host to exchange
  // unique clique ids via HcclRootInfo.
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument("CliqueId is required to create HCCL communicators");
  }
  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Initialize HCCL communicators for %d local devices "
      "(out of %d global devices); size(id)=%zu; fingerprint(id)=%v",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks), ranks.size(),
      clique_key.num_devices(), clique_ids->size(), clique_ids->fingerprint());

  const auto& gpu_config =
      tsl::down_cast<const gpu::GpuCollectives::Config&>(config);
  if (!gpu_config.blocking_communicators && !gpu_config.async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }

  TF_ASSIGN_OR_RETURN(auto stream_executors, GetStreamExecutors(ranks));

  // make_comm returns a new HcclComm.
  auto make_comm = [&](int i) -> absl::StatusOr<HcclComm> {
    absl::Time init_start = absl::Now();
    VLOG(1) << absl::StreamFormat(
        "[%d] [rank=%v] Initialize HCCL communicator for rank %v of %d; "
        "size(id)=%zu; fingerprint(id)=%v",
        DeviceOrdinal(ranks[i]), ranks[i].rank, ranks[i].rank,
        clique_key.num_devices(), clique_ids->size(),
        clique_ids->fingerprint());

    auto* device = tsl::down_cast<gpu::GpuCollectives::Device*>(ranks[i].device);
    TF_RET_CHECK(device != nullptr);
    auto activate_context = device->stream_executor()->Activate();

    std::vector<HcclRootInfo> hccl_root_infos;
    for (const CliqueId& clique_id : clique_ids->data()) {
      TF_ASSIGN_OR_RETURN(hccl_root_infos.emplace_back(),
                          AsHcclRootInfo(clique_id));
    }

    HcclComm comm;
    // HCCL uses HcclCommInitRootInfo for multi-device initialization
    // For now, we use the first root info (support for single-clique case)
    if (hccl_root_infos.size() > 1) {
      // TODO: Support scalable HCCL initialization with multiple root infos
      return Unimplemented("Multiple root infos not yet supported for HCCL");
    } else {
      HcclResult result = HcclCommInitRootInfo(
          clique_key.num_devices(), &hccl_root_infos[0],
          ranks[i].rank.value(), &comm);
      TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclCommInitRootInfo failed"));
    }

    absl::Time init_done = absl::Now();
    VLOG(1) << absl::StreamFormat(
        "[%d] [rank=%v] Initialized HCCL communicator for rank %v of %d in %s",
        DeviceOrdinal(ranks[i]), ranks[i].rank, ranks[i].rank,
        clique_key.num_devices(), absl::FormatDuration(init_done - init_start));

    return comm;
  };

  // Create all communicators. Each communicator is created on its own thread.
  std::vector<std::unique_ptr<Communicator>> comms(ranks.size());
  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateCommunicators",
                                 ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
      pool.Schedule([&, i]() {
        absl::StatusOr<std::unique_ptr<HcclCommunicator>> comm =
            HcclCommunicator::Create(stream_executors[i],
                                     std::bind(make_comm, i), cancel,
                                     gpu_config.async_execution);
        if (!comm.ok()) {
          absl::call_once(once, [&] { status = comm.status(); });
          return;
        }
        comms[i] = *std::move(comm);
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  TF_RETURN_IF_ERROR(status);
  return comms;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
HcclCollectives::SplitCommunicatorsWithCancel(
    absl::Span<const Communicator* const> comms, int32_t color,
    absl::Span<const RankId> keys, const Collectives::Config& config,
    absl::Span<const DeviceRank> ranks,
    std::shared_ptr<xla::gpu::CancellationToken> cancel) {
  auto rank_formatter = [](std::string* str, RankId rank) {
    absl::StrAppend(str, rank.value());
  };

  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Split %d HCCL communicators using color %d and "
      "keys [%s]",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks), comms.size(),
      color, absl::StrJoin(keys, ",", rank_formatter));

  if (keys.size() != comms.size()) {
    return InvalidArgument(
        "Comms and keys must have the same size, but %d != %d", comms.size(),
        keys.size());
  }

  TF_ASSIGN_OR_RETURN(auto stream_executors, GetStreamExecutors(ranks));

  const auto& gpu_config =
      tsl::down_cast<const gpu::GpuCollectives::Config&>(config);

  auto make_comm = [&](int i) -> absl::StatusOr<HcclComm> {
    VLOG(1) << absl::StreamFormat(
        "[%d] [rank=%v] Split HCCL communicator %p with color %d and "
        "key %v",
        DeviceOrdinal(ranks[i]), ranks[i].rank,
        static_cast<const void*>(comms[i]), color, keys[i]);
    
    // HCCL doesn't have a direct CommSplit API like NCCL.
    // We need to create new communicators using HcclCreateSubCommConfig
    // For now, return unimplemented
    return Unimplemented("HCCL SplitCommunicators is not implemented");
  };

  std::vector<std::unique_ptr<Communicator>> split_comms(comms.size());
  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "SplitCommunicators",
                                 comms.size());
    for (size_t i = 0; i < comms.size(); ++i) {
      pool.Schedule([&, i]() {
        absl::StatusOr<std::unique_ptr<HcclCommunicator>> comm =
            HcclCommunicator::Create(stream_executors[i],
                                     std::bind(make_comm, i), cancel,
                                     gpu_config.async_execution);
        if (!comm.ok()) {
          absl::call_once(once, [&] { status = comm.status(); });
          return;
        }
        split_comms[i] = *std::move(comm);
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  TF_RETURN_IF_ERROR(status);
  return split_comms;
}

absl::StatusOr<void*> HcclCollectives::Allocate(uint64_t bytes) {
  // HCCL doesn't have a dedicated memory allocation API.
  // We use Ascend ACL runtime for memory allocation instead.
  void* ptr = nullptr;
  aclError error = aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (error != ACL_SUCCESS) {
    return Internal(
        "Failed to allocate %s (%llu bytes) from device memory: %s",
        tsl::strings::HumanReadableNumBytes(bytes), bytes,
        aclGetRecentErrMsg());
  }
  VLOG(2) << "Allocated device memory " << ptr << " of " << bytes << " bytes";
  return ptr;
}

absl::Status HcclCollectives::Deallocate(void* location) {
  if (location == nullptr) {
    return absl::OkStatus();
  }

  aclError error = aclrtFree(location);
  if (error != ACL_SUCCESS) {
    return Internal(
        "Failed to free device memory at %p: %s",
        location, aclGetRecentErrMsg());
  }

  VLOG(2) << "Deallocated device memory " << location;
  return absl::OkStatus();
}

absl::StatusOr<gpu::GpuCollectives::CliqueIdCallback>
HcclCollectives::InitializeTopology(const Topology& topology) {
  if (topology.num_processes > 1) {
    auto hccl_id_store = std::make_shared<HcclIdStore>(
        topology.process_id, topology.device_to_process,
        std::move(topology.kv_store));
    return [hccl_id_store, this](const CliqueKey& key) {
      return hccl_id_store->GetCliqueIds(key, *this);
    };
  }

  return nullptr;
}

}  // namespace xla::npu

XLA_COLLECTIVES_REGISTER("ASCEND", "hccl", 1,
                         std::make_unique<xla::npu::HcclCollectives>());

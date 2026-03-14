/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/npu/se_ascend_pjrt_client.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/host_to_device_transfer_manager.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/util.h"

namespace xla {

// Implementation of Ascend PjRtClient
class StreamExecutorAscendClient : public PjRtStreamExecutorClient {
 public:
  StreamExecutorAscendClient(
      std::string platform_name,
      LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index,
      std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
      std::unique_ptr<se::DeviceAddressAllocator> allocator,
      std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers);

  ~StreamExecutorAscendClient() override;
};

StreamExecutorAscendClient::StreamExecutorAscendClient(
    std::string platform_name,
    LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index,
    std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
    std::unique_ptr<se::DeviceAddressAllocator> allocator,
    std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers)
    : PjRtStreamExecutorClient(
          std::move(platform_name),
          client,
          std::move(devices),
          process_index,
          std::move(memory_spaces),
          std::move(allocator),
          std::move(host_memory_allocator),
          should_stage_host_to_device_transfers,
          nullptr) {
}

StreamExecutorAscendClient::~StreamExecutorAscendClient() {
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(
    const NpuClientOptions& options) {
  // TODO: Implement actual client creation
  return absl::UnimplementedError("GetPjRtNpuClient not implemented");
}

}  // namespace xla

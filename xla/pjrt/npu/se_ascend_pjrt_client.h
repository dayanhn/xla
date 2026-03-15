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

#ifndef XLA_PJRT_NPU_SE_ASCEND_PJRT_CLIENT_H_
#define XLA_PJRT_NPU_SE_ASCEND_PJRT_CLIENT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/numa.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"

namespace xla {

class StreamExecutorAscendDevice : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorAscendDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state,
                          std::string device_kind, std::string device_vendor,
                          std::string compute_capability, int core_count,
                          int shared_memory_per_block_optin,
                          int local_device_id, int process_index,
                          int process_index_in_partition = 0,
                          int partition_index = 0,
                          int numa_node = tsl::port::kNUMANoAffinity);

  absl::string_view device_vendor() const;

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  absl::Span<int const> coords() const;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

 private:
  std::string device_vendor_;
};

class StreamExecutorAscendHbmMemorySpace : public PjRtStreamExecutorMemorySpace {
 public:
  static constexpr absl::string_view kKind = "device";
  static const int kKindId;

  StreamExecutorAscendHbmMemorySpace(int id, PjRtDevice* device);
};

// A custom PjRtClient that overrides the device assignment method.
class StreamExecutorAscendClient : public xla::PjRtStreamExecutorClient {
 public:
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  StreamExecutorAscendClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index, std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
      std::unique_ptr<se::DeviceAddressAllocator> allocator,
      std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers);

  absl::string_view platform_version() const override;

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  void UpdateGlobalProcessInfo(
      absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override;

  using PjRtStreamExecutorClient::CreateBuffersForAsyncHostToDevice;
  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const PjRtClient::ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadSerialized(
      absl::string_view serialized, std::optional<CompileOptions> options,
      const LoadOptions& load_options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<PjRtStreamExecutorExecutionOutput> RunAsync(
      LocalExecutable& exec, PjRtDevice* device,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> flat_arguments,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> results,
      ExecutableRunOptions run_options_inp, bool parameter_is_tupled_arguments,
      absl::Span<const Shape> executable_parameter_shapes) override;

  absl::Status UpdateCompileOptionsInternal(
      CompileOptions* options, ExecutableExtras* returned_extras,
      bool lookup_addressable_devices) override;
};

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id);

// Public entry point to get an NPU PjRtClient
absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorAscendClient(
    const NpuClientOptions& options);
}  // namespace xla

#endif  // XLA_PJRT_NPU_SE_ASCEND_PJRT_CLIENT_H_
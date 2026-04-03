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
#include "xla/pjrt/npu/ascend_helpers.h"
#include "xla/pjrt/npu/se_ascend_topology_description.h"
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
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
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
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/gpu_constants.h"
namespace xla {

// Implementation of StreamExecutorAscendDevice
StreamExecutorAscendDevice::StreamExecutorAscendDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor,
    std::string compute_capability, int core_count,
    int shared_memory_per_block_optin, int local_device_id, int process_index,
    int process_index_in_partition, int partition_index,
    int numa_node)
    : PjRtStreamExecutorDevice( id, std::move(local_device_state), local_device_id, process_index,
          process_index_in_partition, partition_index, std::move(device_kind)),
      device_vendor_(std::move(device_vendor)) {
}

absl::string_view StreamExecutorAscendDevice::device_vendor() const {
  return device_vendor_;
}

absl::StatusOr<tsl::AllocatorStats> StreamExecutorAscendDevice::GetAllocatorStats() const {
  // Placeholder implementation - return error for now
  return tsl::errors::Unimplemented("GetAllocatorStats not yet implemented for Ascend");
}

absl::Span<int const> StreamExecutorAscendDevice::coords() const {
  // Placeholder implementation - return empty span
  static const std::vector<int> coords = {};
  return absl::MakeSpan(coords);
}

absl::StatusOr<PjRtMemorySpace*> StreamExecutorAscendDevice::default_memory_space() const {
  return memory_space_by_kind_id(StreamExecutorAscendHbmMemorySpace::kKindId);
}

const int StreamExecutorAscendHbmMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(StreamExecutorAscendHbmMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

StreamExecutorAscendHbmMemorySpace::StreamExecutorAscendHbmMemorySpace(
    int id, PjRtDevice* device)
    : PjRtStreamExecutorMemorySpace(id, device, kKind, kKindId) {
}

// Implementation of StreamExecutorAscendClient
StreamExecutorAscendClient::StreamExecutorAscendClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
    std::unique_ptr<se::DeviceAddressAllocator> allocator,
    std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers)
    : PjRtStreamExecutorClient(
          std::move(platform_name), client, std::move(devices), process_index,
          std::move(memory_spaces), std::move(allocator),
          std::move(host_memory_allocator),
          should_stage_host_to_device_transfers,
          nullptr) {
  // Initialize topology description
  int number_of_devices = addressable_devices().size();
  int num_devices_per_host = number_of_devices;
  int number_of_hosts = 1;
  
  topology_.emplace(tsl::Fingerprint64(platform_name), platform_name,
                    number_of_devices, num_devices_per_host, number_of_hosts);
}

absl::string_view StreamExecutorAscendClient::platform_version() const {
  // Placeholder implementation
  return "ascend";
}

std::optional<PjRtPluginAttributes> StreamExecutorAscendClient::plugin_attributes() const {
  PjRtPluginAttributes attrs;
  attrs.pjrt_c_api_major_version = 0;
  attrs.pjrt_c_api_minor_version = 0;
  attrs.attributes["supports_cross_host_transfers"] = PjRtValueType(false);
  return attrs;
}

void StreamExecutorAscendClient::UpdateGlobalProcessInfo(
    absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) {
  // Placeholder implementation
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
StreamExecutorAscendClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  // Placeholder implementation
  return nullptr;
}

absl::StatusOr<xla::DeviceAssignment>
StreamExecutorAscendClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                            num_partitions);
}

absl::StatusOr<Layout> StreamExecutorAscendClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  if (!topology_.has_value()) {
    return absl::FailedPreconditionError("Ascend Topology is missing");
  }
  return topology_->GetDefaultLayout(element_type, dims);
}

absl::StatusOr<const xla::PjRtTopologyDescription*> StreamExecutorAscendClient::GetTopologyDescription() const {
  if (!topology_.has_value()) {
    return absl::FailedPreconditionError("Ascend Topology is missing");
  }
  return &*topology_;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> StreamExecutorAscendClient::LoadSerialized(
    absl::string_view serialized, std::optional<CompileOptions> options,
    const LoadOptions& load_options) {
  // Placeholder implementation
  return nullptr;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> StreamExecutorAscendClient::CompileAndLoad(
    const XlaComputation& computation, CompileOptions options) {
  // Placeholder implementation
  return PjRtStreamExecutorClient::CompileAndLoad(computation, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> StreamExecutorAscendClient::CompileAndLoad(
    mlir::ModuleOp module, CompileOptions options) {
  // Placeholder implementation
  return PjRtStreamExecutorClient::CompileAndLoad(module, options);
}

static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return gpu::kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return gpu::kConstantBufferAlignBytes;
    } else {
      return gpu::kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtStreamExecutorExecutionOutput>
StreamExecutorAscendClient::RunAsync(
    LocalExecutable& exec, PjRtDevice* device,
    absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> flat_arguments,
    absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> results,
    ExecutableRunOptions run_options_inp, bool parameter_is_tupled_arguments,
    absl::Span<const Shape> executable_parameter_shapes) {
#if 1
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(flat_arguments.size());
  for (const Shape& arg_shape : executable_parameter_shapes) {
    argument_shapes.push_back(&arg_shape);
  }

  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      exec.RunHelper(argument_shapes, run_options_inp));
  auto* gpu_exec =
      tensorflow::down_cast<xla::gpu::GpuExecutable*>(exec.executable());
  const ServiceExecutableRunOptions* run_options = &options_and_stream.first;
  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();

  se::StreamExecutor* executor = run_options->stream()->parent();

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();

  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "[", device_ordinal, "] GpuExecutable::ExecuteAsyncOnStreamImpl(",
      gpu_exec->name(), ")"));

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, &gpu::GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(&gpu::GetGpuMutex(executor));
  }

  const gpu::GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(
        globals, gpu_exec->ResolveConstantGlobals(run_options->stream()));
  }

  absl::Span<const BufferAllocation* const> allocations =
      gpu_exec->GetAllocations();

  std::vector<se::DeviceAddressBase> buffers(allocations.size());
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Build buffer allocations"); },
        tsl::profiler::TraceMeLevel::kInfo);
    const int64_t num_buffers = allocations.size();
    for (int64_t i = 0; i < num_buffers; ++i) {
      const BufferAllocation& allocation = *allocations[i];
      se::DeviceAddressBase& buffer = buffers[i];
      if (allocation.is_thread_local()) {
        // buffer = se::DeviceAddressBase{};
      } else if (allocation.is_entry_computation_parameter()) {
        int64_t param_no;
        if (parameter_is_tupled_arguments) {
          // TODO(parkers): Change compiler to not even pretend to read
          // the tuple index tables (also GPU shouldn't tuple ever).
          if (allocation.param_shape_index().empty()) {
            continue;
          }
          param_no = allocation.param_shape_index()[0];
        } else {
          param_no = allocation.parameter_number();
        }
        buffer = tensorflow::down_cast<const xla::PjRtStreamExecutorRawBuffer*>(
                     flat_arguments[param_no].get())
                     ->device_buffer()
                     ->mem();
        if (buffer.is_null() && buffer.size() > 0) {
          return FailedPrecondition(
              "Cannot run XLA computation because pointer to (sub-)buffer at "
              "index %s of parameter %d was null.  All pointers to "
              "(sub-)buffers must not be null, unless the (sub-)buffer has "
              "zero elements.",
              allocation.param_shape_index().ToString(), param_no);
        }
      } else if (allocation.is_constant()) {
        auto it = globals->find(i);
        if (it != globals->end()) {
          buffer = it->second;
        }
      } else {
        // Allocate each allocation that might escape, or is the temp buffer.
        CHECK(allocation.maybe_live_out() ||
              allocation.IsPreallocatedTempBuffer());
        const int64_t buffer_size = allocation.size();
        if (buffer_size > 0) {
          TF_ASSIGN_OR_RETURN(
              se::ScopedDeviceAddress<uint8_t> owning_buffer,
              memory_allocator->Allocate(device_ordinal, buffer_size,
                                         /*retry_on_failure=*/true,
                                         /*memory_space=*/allocation.color()));
          buffer = owning_buffer.Release();
        }
      }
      TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
    }
  }
  xla::gpu::BufferAllocations buffer_allocations(buffers, device_ordinal,
                                                 memory_allocator);
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Buffer allocations: " << buffer_allocations.ToString();

  std::set<se::DeviceAddressBase> buffers_in_result;

  auto set_result = [&](const ShapeIndex& index, int i) -> absl::Status {
    const gpu::GpuExecutable::OutputInfo& output_info =
        gpu_exec->output_info().at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceAddressBase result_buffer;

    XLA_VLOG_DEVICE(4, device_ordinal)
        << "Looking at: allocation " << output_info.allocation_index
        << " @ index: " << index.ToString();

    auto buf =
        tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(results[i].get())
            ->device_buffer();
    if (output_info.alias_config) {
      auto input = tensorflow::down_cast<xla::PjRtStreamExecutorRawBuffer*>(
                       flat_arguments[parameter_is_tupled_arguments
                                          ? allocation->param_shape_index()[0]
                                          : allocation->parameter_number()]
                           .get())
                       ->device_buffer();
      bool is_donated = input == buf;
      if (output_info.alias_config->must_alias() && !is_donated) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (is_donated) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        buffers_in_result.insert(input->mem());
        return absl::OkStatus();
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(gpu_exec->result_shape(), index)
                      .IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        XLA_VLOG_DEVICE(3, device_ordinal)
            << "Using copy-protection: aliasing is specified, but the "
               "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(gpu_exec->result_shape(), index));
        absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return gpu_exec->VerboseAllocationError(allocated_buffer.status());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceAddressBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);
    }
    buffers_in_result.insert(result_buffer);

    RawSEDeviceMemory::ConstructDelayed(
        buf, result_buffer,
        tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
            ->local_device_state(),
        memory_allocator);
    return absl::OkStatus();
  };

  if (gpu_exec->result_shape().IsTuple()) {
    int tuple_count = gpu_exec->result_shape().tuple_shapes().size();
    for (int i = 0; i < tuple_count; ++i) {
      TF_RETURN_IF_ERROR(set_result({i}, i));
    }
  } else {
    TF_RETURN_IF_ERROR(set_result({}, 0));
  }

  TF_RETURN_IF_ERROR(gpu_exec->ExecuteThunks(buffer_allocations, run_options));

  TF_RETURN_IF_ERROR(buffer_allocations.TearDown(buffers_in_result,
                                                 gpu_exec->GetAllocations()));

  std::vector<tsl::AsyncValueRef<RawSEDeviceMemory>> to_be_released;

  return PjRtStreamExecutorExecutionOutput({std::move(to_be_released), {}});
#else      
  // Placeholder implementation
  return PjRtStreamExecutorClient::RunAsync(exec, device, flat_arguments,
                                           results, run_options_inp,
                                           parameter_is_tupled_arguments,
                                           executable_parameter_shapes);
#endif
}

absl::Status StreamExecutorAscendClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras,
    bool lookup_addressable_devices) {
  // Placeholder implementation
  return PjRtStreamExecutorClient::UpdateCompileOptionsInternal(
      options, returned_extras, lookup_addressable_devices);
}

// Implementation of BuildLocalDevices
std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& desc = 
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorAscendDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        desc.name(), desc.device_vendor(),
        "", // compute capability string for Ascend
        desc.core_count(),
        desc.shared_memory_per_block_optin(),
        ordinal_and_device.second->local_device_id().value(),
        node_id,
        desc.numa_node());
    devices.push_back(std::move(device));
  }
  return devices;
}

// Builds a LocalDeviceState for each Ascend device present.
absl::StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}


// Constructs a Ascend device memory allocator to use, according to the allocator
// configuration the client requested.
absl::StatusOr<std::unique_ptr<se::DeviceAddressAllocator>>
GetStreamExecutorAscendDeviceAllocator(
    se::Platform* platform, const NpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices) {
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  switch (allocator_config.kind) {
    case NpuAllocatorConfig::Kind::kDefault:
    case NpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate,
                               allocator_config.npu_system_memory_size,
                               allocator_config.sub_allocator_alloc_visitors,
                               allocator_config.sub_allocator_free_visitors));
        allocators.emplace_back(
            std::move(bfc_allocator),
            ordinal_and_device.second->compute_stream(),
            /*memory_space=*/0); // Default memory space
      }
      break;
    }

    case NpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      if (allocator_config.collective_memory_size != 0) {
        LOG(WARNING)
            << "collective_memory_size is non-zero, but allocator kind is set "
               "to \"platform\". Collective memory will not be allocated.";
      }
      // Returning null will cause the client to use the default backend
      // allocator.
      return nullptr;
  }

  // Add any additional allocators for alternate memory spaces.
  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        CreateCollectiveBFCAllocator(
            ordinal_and_device.second->executor(),
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(
        std::move(collective_bfc_allocator),
        ordinal_and_device.second->compute_stream(),
        /*memory_space=*/1); // Collective memory space
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto host_allocator,
        GetAscendHostAllocator(ordinal_and_device.second->executor()));
    allocators.emplace_back(
        std::move(host_allocator), ordinal_and_device.second->compute_stream(),
        /*memory_space=*/
        static_cast<int>(stream_executor::MemorySpace::kHost));
  }

  return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                  std::move(allocators));
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorAscendClient(
    const NpuClientOptions& options) {
  auto pjrt_platform_name = "ASCEND";

  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetAscendXlaClient(options.platform_name, options.allowed_devices));
  
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
 
  EnableAscendPeerAccess(xla_client->backend().stream_executors());
      
  TF_ASSIGN_OR_RETURN(auto allocator,
                      GetStreamExecutorAscendDeviceAllocator(
                          xla_client->platform(), options.allocator_config,
                          local_device_states));
                      
  std::unique_ptr<HostMemoryAllocator> host_memory_allocator;
  if (options.host_memory_allocator_factory != nullptr) {
    stream_executor::StreamExecutor* const stream_executor = 
        local_device_states.begin()->second->compute_stream()->parent();
    HostMemoryAllocator::Options allocator_options;
    allocator_options.alignment = tsl::Allocator::kAllocatorAlignment;
    allocator_options.map_fn = [stream_executor](void* data, size_t size) {
      // For Ascend, we may not need to register host memory
      // This is a placeholder implementation
      return absl::OkStatus();
    };
    allocator_options.unmap_fn = [stream_executor](void* data) {
      // For Ascend, we may not need to unregister host memory
      // This is a placeholder implementation
      return absl::OkStatus();
    };
    TF_ASSIGN_OR_RETURN(
        host_memory_allocator,
        options.host_memory_allocator_factory(std::move(allocator_options)));
  } else {
    TF_ASSIGN_OR_RETURN(
        auto allocator,
        GetAscendHostAllocator(local_device_states.begin()->second->executor()));
    host_memory_allocator = std::make_unique<BasicHostMemoryAllocator>(
        std::move(allocator), tsl::Allocator::kAllocatorAlignment);
  }
  
  auto devices = BuildLocalDevices(std::move(local_device_states), options.node_id);

  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  for (auto& device : devices) {
    auto memory_space = std::make_unique<StreamExecutorAscendHbmMemorySpace>(
        device->id(), device.get());
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device.get())->AttachMemorySpace(
        memory_space.get(), /*is_default=*/true);
    memory_spaces.push_back(std::move(memory_space));
  }

  return std::make_unique<StreamExecutorAscendClient>(
      pjrt_platform_name, xla_client, std::move(devices),
      options.node_id, std::move(memory_spaces),
      std::move(allocator), std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers);
}

}  // namespace xla
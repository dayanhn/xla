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

#include "xla/stream_executor/ascend/ascend_executor.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/container/flat_hash_map.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/ascend/ascend_context.h"
#include "xla/stream_executor/ascend/ascend_event.h"
#include "xla/stream_executor/ascend/ascend_stream.h"
#include "xla/stream_executor/ascend/scoped_activate_context.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/xla_data.pb.h"
#include "xla/stream_executor/ascend/ascend_context.h"
#include "xla/stream_executor/ascend/scoped_activate_context.h"
#include "xla/util.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"

namespace stream_executor::ascend {

namespace {

// Returns whether peer access is possible between two devices.

// Allocates host memory that can be efficiently transferred to/from the device.
// If numa_node is not kNUMANoAffinity, the memory is allocated on that NUMA node.
absl::StatusOr<void*> HostAllocate(AscendContext* context, int numa_node, uint64_t size) {
#if 0
  if (numa_node != tsl::port::kNUMANoAffinity) {
    // Allocate memory on the specified NUMA node
    auto* buffer = tsl::port::NUMAMalloc(numa_node, size, /* minimum_alignment=*/256);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate host memory of size %d pinned to NUMA node %d",
          size, numa_node));
    }
    // No need to register memory for Ascend, as ACL handles this automatically
    return buffer;
  }
#endif

  ScopedActivateContext activation(context);
  void* buffer = nullptr;
  // Allocate pinned host memory for efficient device access
  aclError error = aclrtMallocHost((void**)&buffer, size);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("aclrtMallocHost failed with ", error));
  }
  if (!buffer && size > 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to allocate pinned host memory of size %d", size));
  }
  return buffer;
}

// Deallocates memory allocated via HostAllocate.
void HostDeallocate(AscendContext* context, int numa_node, void* location, uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    tsl::port::NUMAFree(location, size);
  } else {
    ScopedActivateContext activation(context);
    auto status = aclrtFreeHost(location);
    if (status != ACL_ERROR_NONE) {
      XLA_LOG_DEVICE(ERROR, context->device_ordinal())
          << "failed to free host memory at " << location
          << "; result: " << status;
    }
  }
}

// Allocates host memory and returns a MemoryAllocation wrapping it.
absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    AscendContext* context, int numa_node, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, HostAllocate(context, numa_node, size));
  XLA_VLOG_DEVICE(2, context->device_ordinal())
      << "allocated " << ptr << " for context " << context << " of "
      << size << " bytes of host memory";
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [context, numa_node](void* location, uint64_t size) {
        HostDeallocate(context, numa_node, location, size);
        XLA_VLOG_DEVICE(2, context->device_ordinal())
            << "deallocated host memory at " << location
            << " for context " << context;
      });
}

// Returns whether peer access is possible between two devices.
bool CanEnablePeerAccess(int from_device, int to_device) {
  int32_t result_s2d = -1;
  aclError error = aclrtDeviceCanAccessPeer(&result_s2d, from_device, to_device);
  if (error != ACL_ERROR_NONE || !result_s2d){
    return false;
  }
  return true;
}

// Enables peer access between two contexts.
absl::Status EnablePeerAccess(int from_device, int to_device) {
  aclError error = aclrtSetDevice(from_device);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtSetDevice failed with " << error << ",set device is " << from_device;
  }
  error = aclrtDeviceEnablePeerAccess(to_device, 0);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed with " << error << ", from device " \
               << from_device << " to device " << to_device;
  }
  return absl::OkStatus();
}

}  // namespace



AscendExecutor::~AscendExecutor() {
  // Context is managed by AscendContext::GetContextMap()
}

absl::Status AscendExecutor::Init() {
  // Create Ascend context
  TF_ASSIGN_OR_RETURN(context_, AscendContext::Create(device_ordinal_));

  // Detect peer access capabilities
  uint32_t device_count = 0;
  aclError error = aclrtGetDeviceCount(&device_count);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtGetDeviceCount failed with " << error;
    device_count = 1; // Assume at least one device
  }

  for (int i = 0; i < device_count; ++i) {
    if (i == device_ordinal_) {
      peer_access_cache_[i] = true;
      continue;
    }

    peer_access_cache_[i] = CanEnablePeerAccess(device_ordinal_, i);
  }

  return absl::OkStatus();
}

std::unique_ptr<ActivateContext> AscendExecutor::Activate() {
  return std::make_unique<ScopedActivateContext>(context_);
}

bool AscendExecutor::SynchronizeAllActivity() {
  ScopedActivateContext activation(context_);
  aclError error = aclrtSynchronizeDevice();
  if (error != ACL_ERROR_NONE) {
    XLA_LOG_DEVICE(ERROR, device_ordinal())
        << "aclrtSynchronizeDevice failed with " << error;
    return false;
  }
  return true;
}

absl::StatusOr<DeviceAddressBase> AscendExecutor::GetMemoryRange(
    const DeviceAddressBase& location) const {
  // For Ascend, we'll return the same address and size since we don't have a direct API
  // to get the memory range. This is a simplification.
  return DeviceAddressBase(location.opaque(), location.size());
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>> AscendExecutor::CreateEventBasedTimer(
    Stream* stream, bool use_delay_kernel) {
  // TODO: Implement event-based timer creation
  return absl::UnimplementedError("CreateEventBasedTimer not implemented");
}

absl::StatusOr<DeviceAddressBase> AscendExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  // TODO: Implement symbol retrieval
  return absl::UnimplementedError("GetSymbol not implemented");
}

absl::Status AscendExecutor::SynchronousMemZero(DeviceAddressBase* location,
                                              uint64_t size) {
  ScopedActivateContext activation(context_);
  aclError error = aclrtMemset(location->opaque(), size, 0, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("SynchronousMemZero failed with ", error));
  }
  return absl::OkStatus();
}

absl::Status AscendExecutor::SynchronousMemcpy(DeviceAddressBase* ascend_dst,
                                             const void* host_src, uint64_t size) {
  ScopedActivateContext activation(context_);
  aclError error = aclrtMemcpy(ascend_dst->opaque(), size, host_src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("SynchronousMemcpy H2D failed with ", error));
  }
  XLA_VLOG_DEVICE(2, device_ordinal())
      << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status AscendExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceAddressBase& ascend_src,
                                             uint64_t size) {
  ScopedActivateContext activation(context_);
  aclError error = aclrtMemcpy(host_dst,size, ascend_src.opaque(), size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("SynchronousMemcpy D2H failed with ", error));
  }
  XLA_VLOG_DEVICE(2, device_ordinal()) << "successfully sync memcpy'd d2h of "
                                       << size << " bytes to " << host_dst;
  return absl::OkStatus();
}

void AscendExecutor::DeallocateStream(Stream* stream) {
  AscendStream* ascend_stream = static_cast<AscendStream*>(stream);
  absl::MutexLock l(&alive_ascend_streams_mu_);
  alive_ascend_streams_.erase(ascend_stream->stream_handle());
}

absl::Status AscendExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  AscendExecutor* other_ascend_executor = static_cast<AscendExecutor*>(other);
  int other_device_ordinal = other_ascend_executor->device_ordinal();

  if (other_device_ordinal == device_ordinal()) {
    return absl::OkStatus(); // A device can always access its own memory
  }

  if (!CanEnablePeerAccessTo(other_device_ordinal)) {
    return absl::InternalError(absl::StrCat("Peer access not supported from device ", 
                                           device_ordinal(), " to ", other_device_ordinal));
  }

  return EnablePeerAccess(device_ordinal(), other_device_ordinal);
}

bool AscendExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  AscendExecutor* other_ascend_executor = static_cast<AscendExecutor*>(other);
  return CanEnablePeerAccessTo(other_ascend_executor->device_ordinal());
}

bool AscendExecutor::CanEnablePeerAccessTo(int other_device_ordinal) {
  auto it = peer_access_cache_.find(other_device_ordinal);
  if (it != peer_access_cache_.end()) {
    return it->second;
  }

  // If not in cache, check and add to cache
  bool can_access = CanEnablePeerAccess(device_ordinal_, other_device_ordinal);
  peer_access_cache_[other_device_ordinal] = can_access;
  return can_access;
}

bool AscendExecutor::DeviceMemoryUsage(int64_t* free_out, int64_t* total_out) const {
  size_t device_free;
  size_t device_total;
  aclError error = aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtGetMemInfo failed with " << error;
    return false;
  }
  *free_out = (int64_t)device_free;
  *total_out = (int64_t)device_total;
  return true;
}

absl::StatusOr<std::unique_ptr<Kernel>> AscendExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  // TODO: Implement kernel loading
  return absl::UnimplementedError("LoadKernel not implemented");
}

void AscendExecutor::UnloadKernel(const Kernel* kernel) {
  // TODO: Implement kernel unloading
}

absl::StatusOr<ModuleHandle> AscendExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // TODO: Implement module loading
  return ModuleHandle();  // 返回一个默认构造的ModuleHandle，其id为nullptr
  //return absl::UnimplementedError("LoadModule not implemented");
}

bool AscendExecutor::UnloadModule(ModuleHandle module_handle) {
  // TODO: Implement module unloading
  return false;
}

absl::StatusOr<std::shared_ptr<DeviceAddressBase>> AscendExecutor::CreateOrShareConstant(
    Stream* stream, absl::Span<const uint8_t> content) {
  // TODO: Implement constant creation
  return absl::UnimplementedError("CreateOrShareConstant not implemented");
}

DeviceAddressBase AscendExecutor::Allocate(uint64_t size, int64_t memory_space) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "AscendExecutor::Allocate size: " << size
      << " memory_space: " << memory_space;

  if (memory_space == static_cast<int64_t>(MemorySpace::kCollective)) {
    ScopedActivateContext activation(context_);
    void* result = nullptr;
    // For now, use regular device memory for collective operations
    aclError error = aclrtMalloc(&result, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (error != ACL_ERROR_NONE) {
      XLA_LOG_DEVICE(ERROR, device_ordinal())
          << "Failed to allocate collective memory: " << error;
      return DeviceAddressBase(nullptr, 0);
    }
    XLA_VLOG_DEVICE(1, device_ordinal())
        << "AscendExecutor::Allocate returns " << result;
    return DeviceAddressBase(result, size);
  }

  if (memory_space == static_cast<int64_t>(MemorySpace::kHost)) {
    auto result = HostAllocate(context_, numa_node_, size);
    if (!result.ok()) {
      XLA_LOG_DEVICE(ERROR, device_ordinal())
          << "Failed to allocate host memory: " << result.status();
      return DeviceAddressBase(nullptr, 0);
    }
    XLA_VLOG_DEVICE(1, device_ordinal())
        << "AscendExecutor::Allocate returns " << result.value();
    return DeviceAddressBase(result.value(), size);
  }

  CHECK(memory_space == static_cast<int64_t>(MemorySpace::kDevice) ||
        memory_space == static_cast<int64_t>(MemorySpace::kP2P));

  ScopedActivateContext activation(context_);
  void* result = nullptr;
  aclError error = aclrtMalloc(&result, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (error != ACL_ERROR_NONE) {
    XLA_LOG_DEVICE(ERROR, device_ordinal())
        << "Failed to allocate device memory: " << error;
    return DeviceAddressBase(nullptr, 0);
  }
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "AscendExecutor::Allocate returns " << result;
  return DeviceAddressBase(result, size);
}

void AscendExecutor::Deallocate(DeviceAddressBase* mem) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "AscendExecutor::Deallocate mem: " << mem->opaque();

  // For simplicity, we'll assume all memory is device memory for now
  // In the future, we could add memory space detection
  ScopedActivateContext activation(context_);
  aclError error = aclrtFree(mem->opaque());
  if (error != ACL_ERROR_NONE) {
    XLA_LOG_DEVICE(ERROR, device_ordinal())
        << "failed to free device memory at " << mem->opaque()
        << "; result: " << error;
  } else {
    XLA_VLOG_DEVICE(2, device_ordinal())
        << "deallocated " << mem->opaque() << " for context " << context_;
  }
}

blas::BlasSupport* AscendExecutor::AsBlas() {
  // TODO: Implement BLAS support
  return nullptr;
}

fft::FftSupport* AscendExecutor::AsFft() {
  // TODO: Implement FFT support
  return nullptr;
}

dnn::DnnSupport* AscendExecutor::AsDnn() {
  // TODO: Implement DNN support
  return nullptr;
}

absl::StatusOr<std::unique_ptr<Event>> AscendExecutor::CreateEvent() {
  return AscendEvent::Create(this, /*allow_timing=*/false);
}

absl::StatusOr<std::unique_ptr<Stream>> AscendExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto stream, AscendStream::Create(this, priority));
  absl::MutexLock l(&alive_ascend_streams_mu_);
  alive_ascend_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> AscendExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode) {
  // TODO: Implement command buffer creation
  return absl::UnimplementedError("CreateCommandBuffer not implemented");
}

absl::StatusOr<std::unique_ptr<DeviceDescription>> AscendExecutor::CreateDeviceDescription() const {
  DeviceDescription desc;

  // Set device address bits (Ascend devices support 64-bit addresses)
  desc.set_device_address_bits(64);

  std::string socName = aclrtGetSocName();
  desc.set_name(absl::StrCat(socName+"-", device_ordinal()));

  int64_t total_memory = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, &total_memory);
  desc.set_device_memory_size(total_memory);

  // Set device vendor
  desc.set_device_vendor("Huawei");

  // Set other default values
  int64_t aicore_num = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_num);
  desc.set_core_count(aicore_num);

  int64_t l2_size = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_L2_CACHE_SIZE, &l2_size);
  desc.set_l2_cache_size(l2_size);


  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> AscendExecutor::HostMemoryAllocate(
    uint64_t size) {
  return AllocateHostMemory(context_, numa_node_, size);
}

bool AscendExecutor::HostMemoryRegister(void* location, uint64_t size) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "Called StreamExecutor::HostMemoryRegister(data=" << location << ")";
  // For Ascend, we don't need to register host memory explicitly
  // ACL handles memory registration automatically
  return true;
}

bool AscendExecutor::HostMemoryUnregister(void* location) {
  XLA_VLOG_DEVICE(1, device_ordinal())
      << "Called StreamExecutor::HostUnregister(data=" << location << ")";
  // For Ascend, we don't need to unregister host memory explicitly
  // ACL handles memory unregistration automatically
  return true;
}

absl::StatusOr<MemorySpace> AscendExecutor::GetPointerMemorySpace(const void* ptr) {
  // TODO: Implement memory space detection
  return absl::UnimplementedError("GetPointerMemorySpace not implemented");
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
AscendExecutor::CreateMemoryAllocator(MemorySpace type) {
  if (type == MemorySpace::kHost) {
    return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
      return AllocateHostMemory(context_, numa_node_, size);
    });
  }

  if (type == MemorySpace::kUnified) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          std::unique_ptr<ActivateContext> activation = Activate();
          void* result = nullptr;
          // Allocate unified memory (device memory that is accessible from host)
          aclError error = aclrtMalloc(&result, size, ACL_MEM_MALLOC_HUGE_FIRST);
          if (error != ACL_ERROR_NONE) {
            return absl::InternalError(absl::StrCat("aclrtMalloc failed with ", error));
          }
          XLA_VLOG_DEVICE(2, device_ordinal())
              << "allocated " << result << " for context " << context()
              << " of " << size << " bytes in unified memory";
          return std::make_unique<GenericMemoryAllocation>(
              result, size, [this](void* location, uint64_t size) {
                std::unique_ptr<ActivateContext> activation = Activate();
                auto status = aclrtFree(location);
                if (status != ACL_ERROR_NONE) {
                  XLA_LOG_DEVICE(ERROR, device_ordinal())
                      << "failed to free unified memory at " << location
                      << "; result: " << status;
                } else {
                  XLA_VLOG_DEVICE(2, device_ordinal())
                      << "deallocated unified memory at " << location
                      << " for context " << context();
                }
              });
        });
  }

  if (type == MemorySpace::kCollective) {
    // TODO: Implement collective memory allocation for Ascend
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          std::unique_ptr<ActivateContext> activation = Activate();
          void* result = nullptr;
          // For now, use regular device memory for collective operations
          aclError error = aclrtMalloc(&result, size, ACL_MEM_MALLOC_HUGE_FIRST);
          if (error != ACL_ERROR_NONE) {
            return absl::InternalError(absl::StrCat("aclrtMalloc failed with ", error));
          }
          XLA_VLOG_DEVICE(2, device_ordinal())
              << "allocated " << result << " for context " << context()
              << " of " << size << " bytes of collective memory";
          return std::make_unique<GenericMemoryAllocation>(
              result, size, [this](void* location, uint64_t size) {
                std::unique_ptr<ActivateContext> activation = Activate();
                auto status = aclrtFree(location);
                if (status != ACL_ERROR_NONE) {
                  XLA_LOG_DEVICE(ERROR, device_ordinal())
                      << "failed to free collective memory at " << location
                      << "; result: " << status;
                } else {
                  XLA_VLOG_DEVICE(2, device_ordinal())
                      << "deallocated collective memory at " << location
                      << " for context " << context();
                }
              });
        });
  }

  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

}  // namespace stream_executor::ascend
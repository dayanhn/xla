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

// Public entry point to get an NPU PjRtClient
absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(
    const NpuClientOptions& options);
}  // namespace xla

#endif  // XLA_PJRT_NPU_SE_ASCEND_PJRT_CLIENT_H_

/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/pjrt/npu/se_ascend_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

/*static*/ void StreamExecutorAscendTopologyDescription::SetupDeviceDescription(
    PjRtStreamExecutorDeviceDescription& description,
    const std::string& device_vendor, const std::string& compute_capability,
    int core_count, int64_t shared_memory_per_block_optin,
    int partition_index) {
  std::vector<int64_t> v_coords(description.coords().begin(),
                                description.coords().end());

  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes = {
      {"coords", xla::PjRtDeviceAttribute(v_coords)},
      {"device_vendor", device_vendor},
      {"partition_index", static_cast<int64_t>(partition_index)},
      {"compute_capability", xla::PjRtDeviceAttribute(compute_capability)},
      {"shared_memory_per_block_optin", shared_memory_per_block_optin},
      {"core_count", static_cast<int64_t>(core_count)},
  };
  description.SetAttributes(std::move(attributes));
  description.SetToString(absl::StrFormat(
      "StreamExecutorAscendDevice(device_kind=%s, id=%i, process_index=%i, "
      "partition_index=%i))",
      description.device_kind(), description.id(), description.process_index(),
      partition_index));
  description.SetDebugString(absl::StrFormat(
      "%s_%i(process=%i,(%i))", description.device_kind(), description.id(),
      description.process_index(), v_coords.empty() ? 0 : v_coords[0]));
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
StreamExecutorAscendTopologyDescription::DeviceDescriptions() const {
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
  if (number_of_devices_ <= 0) {
    return devices;
  }
  devices.reserve(number_of_devices_);
  for (int device_id = 0; device_id < number_of_devices_; ++device_id) {
    devices.push_back(CreateDeviceDescription(device_id));
  }
  return devices;
}

std::unique_ptr<PjRtStreamExecutorDeviceDescription>
StreamExecutorAscendTopologyDescription::CreateDeviceDescription(
    int device_id) const {
  const int32_t num_devices_per_process = num_devices_per_host_;
  const int32_t num_processes_per_partition = 1;
  
  const int local_device_id = num_devices_per_process == -1 ? 0 : (device_id % num_devices_per_process);
  const int process_index = num_devices_per_process == -1 ? 0 : (device_id / num_devices_per_process);
  const int process_index_in_partition = process_index == -1 ? 0 : (process_index % num_processes_per_partition);
  const int partition_index = num_processes_per_partition == -1 ? 0 : (process_index / num_processes_per_partition);
  
  auto description = std::make_unique<PjRtStreamExecutorDeviceDescription>(
      device_id, local_device_id, process_index, process_index_in_partition,
      partition_index, std::string(platform_version()));
  
  if (target_config_.has_value()) {
    std::string compute_capability = "<unknown compute-capability>";
    std::string device_vendor = "<unknown device vendor>";
    
    StreamExecutorAscendTopologyDescription::SetupDeviceDescription(
        *description, device_vendor, compute_capability,
        0, 0, /*partition_index=*/0);
  }
  return description;
}

absl::StatusOr<std::string> StreamExecutorAscendTopologyDescription::Serialize()
    const {
  // Placeholder implementation - return error for now
  return tsl::errors::Unimplemented("Serialize not yet implemented for Ascend");
}

absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
StreamExecutorAscendTopologyDescription::
    ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
        xla::PjRtGlobalDeviceId device_id) const {
  if (device_id.value() < 0 || device_id.value() >= number_of_devices_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Device id ", device_id.value(), " is out of range [0, ",
                     number_of_devices_, ")"));
  }
  auto device_desc = CreateDeviceDescription(device_id.value());
  const auto& coords = device_desc->coords();
  if (coords.size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Ascend topology must have 3 dimensions, but got ", coords.size()));
  }
  return std::make_pair(PjRtDeviceDimensions{coords[0], coords[1], coords[2]},
                        0);
}

absl::StatusOr<Layout> StreamExecutorAscendTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  Layout layout = LayoutUtil::GetWithDefaultLayout(shape).layout();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    layout.set_element_size_in_bits(primitive_util::BitWidth(element_type));
  }
  return layout;
}

absl::StatusOr<xla::PjRtTopologyDescriptionProto>
StreamExecutorAscendTopologyDescription::ToProto() const {
  PjRtTopologyDescriptionProto proto;
  proto.set_platform_id(platform_id());
  proto.set_platform_name(platform_name());
  proto.set_platform_version(platform_version());
  proto.set_is_subslice_topology(is_subslice_topology());
  return proto;
}

absl::StatusOr<std::unique_ptr<StreamExecutorAscendTopologyDescription>>
StreamExecutorAscendTopologyDescription::FromProto(
    const xla::PjRtTopologyDescriptionProto& proto) {
  // Placeholder implementation - return error for now
  return tsl::errors::Unimplemented("FromProto not yet implemented for Ascend");
}

}  // namespace xla

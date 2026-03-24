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

#include "xla/pjrt/c/pjrt_c_api_ascend_internal.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/npu/se_ascend_pjrt_client.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"
#include "xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/service/ascend/ffi/ascend_ffi.h"

namespace pjrt {
namespace ascend_plugin {

namespace {

// Create Ascend PJRT Client
absl::StatusOr<std::unique_ptr<xla::PjRtClient>> CreateAscendClient(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& options) {
  xla::NpuClientOptions npu_options;
  
  // Parse visible_devices option if provided
  if (auto it = options.find("visible_devices"); it != options.end()) {
    if (auto* devices = std::get_if<std::vector<int64_t>>(&it->second)) {
      std::set<int> device_set(devices->begin(), devices->end());
      npu_options.allowed_devices = device_set;
    }
  }
  
  // Create the XLA NPU PJRT client
  return xla::GetXlaPjrtNpuClient(npu_options);
}

}  // namespace

const PJRT_Api* GetAscendPjrtApi() {
  static auto* api = []() -> const PJRT_Api* {
    // Register Ascend FFI handlers
    xla::ffi::RegisterAscendFfiHandlers();
    
    // Create the PJRT API wrapper
    auto* api = new PJRT_Api();
    pjrt::InitializePjrtApi(api);
    
    // Override the Client_Create function
    api->PJRT_Client_Create = [](PJRT_Client_Create_Args* args) -> PJRT_Error* {
      PJRT_RETURN_IF_ERROR(pjrt::ValidateStructSize(
          "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
          args->struct_size));
      
      // Convert create options
      auto options = pjrt::ConvertFromPjRtNamedValueList(
          args->create_options, args->num_options);
      
      // Create Ascend client
      auto client_or = CreateAscendClient(options);
      if (!client_or.ok()) {
        return new PJRT_Error{client_or.status()};
      }
      
      // Create the PJRT client wrapper
      auto* client = new pjrt::CApiClient(args->client);
      client->SetPjrtClient(std::move(*client_or));
      return nullptr;
    };
    
    return api;
  }();
  
  return api;
}

}  // namespace ascend_plugin
}  // namespace pjrt

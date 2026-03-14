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


#include <iostream>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "xla/pjrt/npu/se_ascend_pjrt_client.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"
#include "xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client.h"
#include "xla/client/local_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

// 创建Ascend PJRT客户端
std::unique_ptr<PjRtClient> Compiler_CreateAscendClient() {
  xla::NpuClientOptions options;
  // 可按需调整 options，例如选择特定设备集合：
  // options.allowed_devices = std::set<int>({0});

  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or = xla::GetXlaPjrtNpuClient(options);
  if (!client_or.ok()) {
    std::cerr << "Failed to create Ascend client: " 
              << client_or.status().ToString() << std::endl;
    return nullptr;
  }
  return std::move(*client_or);
}



}  // namespace xla

int main(int argc, char** argv) {
  // 创建Ascend客户端
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or = xla::Compiler_CreateAscendClient();
  if (!client_or.ok()) {
    std::cerr << "Failed to create Ascend client: " 
              << client_or.status().ToString() << std::endl;
    return 1;
  }else{
    std::cout << "Ascend client created successfully.\n";
  }
  return 0;
}
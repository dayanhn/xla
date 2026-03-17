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
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/types.h"
#include "xla/pjrt/npu/se_ascend_pjrt_client.h"
#include "xla/pjrt/plugin/xla_npu/npu_client_options.h"
#include "xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client.h"
#include "xla/client/local_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/client/local_client.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// 从文件读取HloModule内容
std::unique_ptr<HloModule> Compiler_LoadHloModuleFromFile(
    absl::string_view file_path) {
  // 读取文件内容到字符串
  std::string hlo_text;
  absl::Status status = tsl::ReadFileToString(
      tsl::Env::Default(), std::string(file_path), &hlo_text);
  if (!status.ok()) {
    std::cerr << "Failed to read HloModule from " << file_path 
              << ": " << status.ToString() << std::endl;
    return nullptr;
  }

  std::cout << "Loaded HloModule from " << file_path << " successfully.\n";

  // 解析HloModule
  HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);
  // 加载全局调试选项，包括从环境变量XLA_FLAGS解析的选项
  config.set_debug_options(GetDebugOptionsFromFlags());
  
  // 解析HloModule内容:xla/hlo/parser/hlo_parser.cc
  absl::StatusOr<std::unique_ptr<HloModule>> module_or= ParseAndReturnUnverifiedModule(hlo_text, config);
  if (!module_or.ok()) {
    std::cerr << "Failed to load HloModule: " 
              << module_or.status().ToString() << std::endl;
    return nullptr;
  }

  return std::move(*module_or);
}

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

int Compiler_GetAscendBackend(std::unique_ptr<PjRtClient> &client, 
                  xla::Backend* &backend, 
                  se::StreamExecutor* &executor, 
                  int64_t &slice_size) {
  // 将PjRtClient转换为 StreamExecutorAscendClient
  xla::StreamExecutorAscendClient* gpu_client = 
      dynamic_cast<xla::StreamExecutorAscendClient*>(client.get());
  if (!gpu_client) {
    std::cerr << "Failed to cast PjRtClient to StreamExecutorAscendClient" << std::endl;
    return 1;
  }
  
  // 获取LocalClient
  xla::LocalClient* local_client = gpu_client->client();
  if (!local_client) {
    std::cerr << "Failed to get LocalClient" << std::endl;
    return 1;
  }
  
  // 获取Backend
  backend = local_client->mutable_backend();
  if (!backend) {
    std::cerr << "Failed to get Backend" << std::endl;
    return 1;
  }
  
  executor = backend->default_stream_executor();

  return 0;
}



}  // namespace xla

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlomodule_file.mlir>" << std::endl;
    return 1;
  }
  std::string program_path = argv[1];
  // 加载HloModule
  std::unique_ptr<xla::HloModule> hlo_module = xla::Compiler_LoadHloModuleFromFile(program_path);
  if(!hlo_module){ return 1;}

  // 创建Ascend客户端
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or = xla::Compiler_CreateAscendClient();
  if (!client_or.ok()) {
    std::cerr << "Failed to create Ascend client: " 
              << client_or.status().ToString() << std::endl;
    return 1;
  }else{
    std::cout << "Ascend client created successfully.\n";
  }

  std::unique_ptr<xla::PjRtClient> client = std::move(*client_or);

  // 获取Backend
  xla::Backend* backend = nullptr;
  xla::se::StreamExecutor* executor = nullptr;

  xla::Compiler::CompileOptions compile_options;
  if(xla::Compiler_GetAscendBackend(client, backend, executor, compile_options.slice_size) != 0){
    return 1;
  }

  std::cout << "Running Backend..." << std::endl;
  absl::StatusOr<std::unique_ptr<xla::Executable>> executable_or = 
      backend->compiler()->RunBackend(std::move(hlo_module), executor, compile_options);
  if (!executable_or.ok()) {
    std::cerr << "Failed to run Backend: " 
              << executable_or.status().ToString() << std::endl;
    return 1;
  }else{
    std::cout << "Successfully run Backend" << std::endl;
  }

  return 0;
}
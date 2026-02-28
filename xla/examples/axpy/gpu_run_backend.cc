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
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
//#include "xla/tests/literal_test_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/client/local_client.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/debug_options_flags.h"
#include "xla/service/dump.h"

namespace xla {

// 从文件读取HloModule内容并解析
absl::StatusOr<std::unique_ptr<HloModule>> ParseHloModule(
    absl::string_view hlo_text) {
  HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);
  // 加载全局调试选项，包括从环境变量XLA_FLAGS解析的选项
  config.set_debug_options(GetDebugOptionsFromFlags());
  
  // 解析HloModule内容:xla/hlo/parser/hlo_parser.cc
  return ParseAndReturnUnverifiedModule(hlo_text, config);
}

// 从文件读取HloModule内容
absl::StatusOr<std::unique_ptr<HloModule>> LoadHloModuleFromFile(
    absl::string_view file_path) {
  // 读取文件内容到字符串
  std::string hlo_text;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(file_path), &hlo_text));

  std::cout << "Loaded HloModule from " << file_path << " successfully.\n";

  // 解析HloModule
  return ParseHloModule(hlo_text);
}


// 创建GPU PJRT客户端
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient() {
  xla::GpuClientOptions options;
  // 可按需调整 options，例如选择特定设备集合：
  // options.allowed_devices = std::set<int>({0});
  return xla::GetXlaPjrtGpuClient(options);
}


// 主函数
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlomodule_file.mlir>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  
  // 加载HloModule
  absl::StatusOr<std::unique_ptr<xla::HloModule>> module_or = LoadHloModuleFromFile(program_path);
  if (!module_or.ok()) {
    std::cerr << "Failed to load HloModule: " 
              << module_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<xla::HloModule> hlo_module = std::move(*module_or);

  // 创建GPU客户端:xla/pjrt/gpu/se_gpu_pjrt_client.cc
  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or = CreateGpuClient();
  if (!client_or.ok()) {
    std::cerr << "Failed to create GPU client: " 
              << client_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtClient> client = std::move(*client_or);

  // 将PjRtClient转换为StreamExecutorGpuClient
  xla::StreamExecutorGpuClient* gpu_client = 
      dynamic_cast<xla::StreamExecutorGpuClient*>(client.get());
  if (!gpu_client) {
    std::cerr << "Failed to cast PjRtClient to StreamExecutorGpuClient" << std::endl;
    return 1;
  }
  
  // 获取LocalClient
  xla::LocalClient* local_client = gpu_client->client();
  if (!local_client) {
    std::cerr << "Failed to get LocalClient" << std::endl;
    return 1;
  }
  
  // 获取Backend
  xla::Backend* backend = local_client->mutable_backend();
  if (!backend) {
    std::cerr << "Failed to get Backend" << std::endl;
    return 1;
  }
  
  se::StreamExecutor* executor = backend->default_stream_executor();
  
  std::cout << "Running Backend..." << std::endl;
  xla::Compiler::CompileOptions compile_options;
  // 获取拓扑描述
  absl::StatusOr<const xla::PjRtTopologyDescription*> topology_desc_or = gpu_client->GetTopologyDescription();
  if (topology_desc_or.ok()) {
    const xla::PjRtTopologyDescription* topology_desc = *topology_desc_or;
    // 转换为 StreamExecutorGpuTopologyDescription
    const xla::StreamExecutorGpuTopologyDescription* gpu_topology_desc = 
        dynamic_cast<const xla::StreamExecutorGpuTopologyDescription*>(topology_desc);
    if (gpu_topology_desc) {
      compile_options.slice_size = gpu_topology_desc->gpu_topology().slice_size();
    }
  }

  absl::StatusOr<std::unique_ptr<Executable>> executable_or = 
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

}  // namespace xla

// 主函数，调用xla命名空间中的main函数
int main(int argc, char** argv) {
  return xla::main(argc, argv);
}
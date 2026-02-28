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

#include "xla/examples/axpy/compiler_utils.h"

#include <iostream>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/client/local_client.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/env.h"

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

// 创建GPU PJRT客户端
std::unique_ptr<PjRtClient> Compiler_CreateGpuClient() {
  xla::GpuClientOptions options;
  // 可按需调整 options，例如选择特定设备集合：
  // options.allowed_devices = std::set<int>({0});

  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or = xla::GetXlaPjrtGpuClient(options);
  if (!client_or.ok()) {
    std::cerr << "Failed to create GPU client: " 
              << client_or.status().ToString() << std::endl;
    return nullptr;
  }
  return std::move(*client_or);
}

// 获取GPU后端信息
int Compiler_GetGpuBackend(std::unique_ptr<PjRtClient> &client, 
                  xla::Backend* &backend, 
                  se::StreamExecutor* &executor, 
                  int64_t &slice_size) {
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
  backend = local_client->mutable_backend();
  if (!backend) {
    std::cerr << "Failed to get Backend" << std::endl;
    return 1;
  }
  
  executor = backend->default_stream_executor();

  // 获取拓扑描述
  absl::StatusOr<const xla::PjRtTopologyDescription*> topology_desc_or = gpu_client->GetTopologyDescription();
  if (topology_desc_or.ok()) {
    const xla::PjRtTopologyDescription* topology_desc = *topology_desc_or;
    // 转换为 StreamExecutorGpuTopologyDescription
    const xla::StreamExecutorGpuTopologyDescription* gpu_topology_desc = 
        dynamic_cast<const xla::StreamExecutorGpuTopologyDescription*>(topology_desc);
    if (gpu_topology_desc) {
      slice_size = gpu_topology_desc->gpu_topology().slice_size();
    }
  }
  return 0;
}

// 读取StableHLO文件并解析为MLIR模块
mlir::OwningOpRef<mlir::ModuleOp> Compiler_LoadStableHloProgram(
    absl::string_view program_path) {
  // 注册必要的MLIR dialects
  // Context 和 registry 必须在 module 使用期间保持存活。
  // 将它们设为 static，以便在程序整个生命周期内有效，避免返回
  // 的 ModuleOp 持有已销毁的 context 导致悬空指针和段错误。
  static mlir::DialectRegistry registry;
  static mlir::MLIRContext* context = nullptr;
  static bool initialized = false;
  if (!initialized) {
    registry.insert<mlir::func::FuncDialect>();
    mlir::stablehlo::registerAllDialects(registry);
    context = new mlir::MLIRContext();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
    initialized = true;
  }

  // 读取StableHLO程序到字符串
  std::string program_string;
  if(absl::OkStatus() != tsl::ReadFileToString(
      tsl::Env::Default(), std::string(program_path), &program_string)){
    std::cerr << "Failed to read StableHLO program from " << program_path << std::endl;
    return nullptr;
  }

  //std::cout << "Loaded StableHLO program from " << program_path << ":\n"
  //          << program_string << std::endl;
  std::cout << "Loaded StableHLO program from " << program_path << " successfully.\n";

  return mlir::parseSourceString<mlir::ModuleOp>(program_string, context);
  // 加载StableHLO程序
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> program_or = mlir::parseSourceString<mlir::ModuleOp>(program_string, context);
  if (!program_or.ok()) {
    std::cerr << "Failed to load StableHLO program: " 
              << program_or.status().ToString() << std::endl;
    return nullptr;
  }
  return std::move(*program_or);
}

// 将StableHLO转换为HloModule
std::unique_ptr<xla::HloModule> Compiler_GetHloModule(
    mlir::OwningOpRef<mlir::ModuleOp> &program) { 
  // 将StableHLO转换为HloModule并打印计算图
  std::cout << "Converting StableHLO to HloModule..." << std::endl;
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_or = 
      xla::ConvertStablehloToHlo(*program);
  if (!hlo_module_or.ok()) {
    std::cerr << "Failed to convert StableHLO to HloModule: " 
              << hlo_module_or.status().ToString() << std::endl;
    return nullptr;
  }
  std::unique_ptr<xla::HloModule> hlo_module = std::move(*hlo_module_or);
  
  // 打印HloModule的计算图
  std::cout << "\nHloModule computation graph:" << std::endl;
  std::cout << hlo_module->ToString() << std::endl;
  return std::move(hlo_module);
}

int Compiler_GetGpuDeviceInfo(std::unique_ptr<PjRtClient> &client, se::DeviceDescription &device_info) {
  // 获取第一个GPU设备
  if (client->devices().empty()) {
    std::cerr << "No GPU devices found" << std::endl;
    return 1;
  }
  PjRtDevice* device = client->devices()[0];

  // 从设备获取se::DeviceDescription
  auto* stream_executor_device = dynamic_cast<PjRtStreamExecutorDevice*>(device);
  if (!stream_executor_device) {
    return 1;
  }
  absl::StatusOr<LocalDeviceState*> local_device_state_or = stream_executor_device->GetLocalDeviceState();
  if (!local_device_state_or.ok()) {
    std::cerr << "Failed to get LocalDeviceState: " 
              << local_device_state_or.status().ToString() << std::endl;
    return 1;
  }
  LocalDeviceState* local_device_state = *local_device_state_or;
  se::StreamExecutor* executor = local_device_state->executor();
  device_info = executor->GetDeviceDescription();
  return 0;
}


}  // namespace xla
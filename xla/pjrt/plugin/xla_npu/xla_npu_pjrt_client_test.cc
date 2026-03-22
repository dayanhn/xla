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
#include "xla/service/ascend/ffi/ascend_ffi.h"

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
  options.allowed_devices = std::set<int>({6,7});

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

int execute_gelu_program(std::unique_ptr<PjRtClient> &client,std::unique_ptr<PjRtLoadedExecutable> &executable) { 
  // 构造输入参数
  std::cout << "Creating input parameters..." << std::endl;
  
  // 创建第一个输入：32x32的矩阵，值全为2.0
  auto x_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {32, 32});
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      x_literal.Set({i, j}, -2.0f);
    }
  }

  // 获取设备内存空间（使用第一个可用设备）
  PjRtDevice* device = client->devices()[0];
  absl::StatusOr<PjRtMemorySpace*> device_memory_space = 
      device->default_memory_space();
  if (!device_memory_space.ok()) {
    std::cerr << "Failed to get device memory space: " 
              << device_memory_space.status().ToString() << std::endl;
    return 1;
  }

  // 将输入转换为设备缓冲区
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> x = 
      client->BufferFromHostLiteral(x_literal, *device_memory_space);
  if (!x.ok()) {
    std::cerr << "Failed to create x buffer: " 
              << x.status().ToString() << std::endl;
    return 1;
  }

  // 执行计算
  std::cout << "Executing computation on Ascend..." << std::endl;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> result = 
      executable->Execute({{x->get()}}, {});
  if (!result.ok()) {
    std::cerr << "Failed to execute computation: " 
              << result.status().ToString() << std::endl;
    return 1;
  }

  // 获取结果并转换为字面量
  std::cout << "Getting result..." << std::endl;
  absl::StatusOr<std::shared_ptr<Literal>> result_or = result->at(0).at(0)->ToLiteralSync();
  if (!result_or.ok()) {
    std::cerr << "Failed to get result literal: " 
              << result_or.status().ToString() << std::endl;
    return 1;
  }
  std::shared_ptr<Literal> result_literal = *result_or;

  // 输出结果
  std::cout << "Computation output: " << *result_literal << std::endl;
  return 0;
}

int execute_matmul_program(std::unique_ptr<PjRtClient> &client,std::unique_ptr<PjRtLoadedExecutable> &executable) { 
  // 构造输入参数
  std::cout << "Creating input parameters for matmul..." << std::endl;
  
  // 创建第一个输入：16x16的矩阵，值全为2.0
  auto a_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {16, 16});
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a_literal.Set({i, j}, 2.0f);
    }
  }

  // 创建第二个输入：16x16的矩阵，值全为2.0
  auto b_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {16, 16});
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      b_literal.Set({i, j}, 2.0f);
    }
  }

  // 获取设备内存空间（使用第一个可用设备）
  PjRtDevice* device = client->devices()[0];
  absl::StatusOr<PjRtMemorySpace*> device_memory_space = 
      device->default_memory_space();
  if (!device_memory_space.ok()) {
    std::cerr << "Failed to get device memory space: " 
              << device_memory_space.status().ToString() << std::endl;
    return 1;
  }

  // 将输入转换为设备缓冲区
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> a = 
      client->BufferFromHostLiteral(a_literal, *device_memory_space);
  if (!a.ok()) {
    std::cerr << "Failed to create a buffer: " 
              << a.status().ToString() << std::endl;
    return 1;
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> b = 
      client->BufferFromHostLiteral(b_literal, *device_memory_space);
  if (!b.ok()) {
    std::cerr << "Failed to create b buffer: " 
              << b.status().ToString() << std::endl;
    return 1;
  }

  // 执行计算
  std::cout << "Executing matmul computation on Ascend..." << std::endl;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> result = 
      executable->Execute({{a->get(), b->get()}}, {});
  if (!result.ok()) {
    std::cerr << "Failed to execute matmul computation: " 
              << result.status().ToString() << std::endl;
    return 1;
  }

  // 获取结果并转换为字面量
  std::cout << "Getting matmul result..." << std::endl;
  absl::StatusOr<std::shared_ptr<Literal>> result_or = result->at(0).at(0)->ToLiteralSync();
  if (!result_or.ok()) {
    std::cerr << "Failed to get matmul result literal: " 
              << result_or.status().ToString() << std::endl;
    return 1;
  }
  std::shared_ptr<Literal> result_literal = *result_or;

  // 输出结果
  std::cout << "Matmul computation output: " << *result_literal << std::endl;
  return 0;
}


}  // namespace xla

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlomodule_file.mlir>" << std::endl;
    return 1;
  }
  std::string program_path = argv[1];
  
  // 显式注册Ascend FFI handlers
  xla::ffi::RegisterAscendFfiHandlers();
  
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

  // 使用HloModule进行编译
  std::cout << "Compiling HloModule for GPU..." << std::endl;
  xla::XlaComputation computation(hlo_module->ToProto());
  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_run_backend_only(true);
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or = 
      client->CompileAndLoad(computation, compile_options);
  if (!executable_or.ok()) {
    std::cerr << "Failed to compile HloModule: " 
              << executable_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<xla::PjRtLoadedExecutable> executable = std::move(*executable_or);

  std::cout << "Executable created with " << executable->addressable_devices().size() 
            << " devices." << std::endl;

  // 测试 GELU 算子
  // xla::execute_gelu_program(client, executable);
  
  // 测试 Matmul 算子
  xla::execute_matmul_program(client, executable);

  return 0;
}
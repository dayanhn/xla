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

namespace xla {

// 读取StableHLO文件并解析为MLIR模块
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadStableHloProgram(
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
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(program_path), &program_string));

  //std::cout << "Loaded StableHLO program from " << program_path << ":\n"
  //          << program_string << std::endl;
  std::cout << "Loaded StableHLO program from " << program_path << " successfully.\n";

  return mlir::parseSourceString<mlir::ModuleOp>(program_string, context);
}

// 创建GPU PJRT客户端
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateGpuClient() {
  xla::GpuClientOptions options;
  // 可按需调整 options，例如选择特定设备集合：
  // options.allowed_devices = std::set<int>({0});
  return xla::GetXlaPjrtGpuClient(options);
}

int execute_axpy_program(std::unique_ptr<PjRtClient> &client,std::unique_ptr<PjRtLoadedExecutable> &executable) { 
  // 构造输入参数
  std::cout << "Creating input parameters..." << std::endl;
  auto alpha_literal = xla::LiteralUtil::CreateR0<float>(3.14f);
  auto x_literal = xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto y_literal = xla::LiteralUtil::CreateR1<float>({10.5f, 20.5f, 30.5f, 40.5f});

  std::cout << "Computation inputs:" << std::endl;
  std::cout << "\talpha:" << alpha_literal << std::endl;
  std::cout << "\tx:" << x_literal << std::endl;
  std::cout << "\ty:" << y_literal << std::endl;

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
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> alpha = 
      client->BufferFromHostLiteral(alpha_literal, *device_memory_space);
  if (!alpha.ok()) {
    std::cerr << "Failed to create alpha buffer: " 
              << alpha.status().ToString() << std::endl;
    return 1;
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> x = 
      client->BufferFromHostLiteral(x_literal, *device_memory_space);
  if (!x.ok()) {
    std::cerr << "Failed to create x buffer: " 
              << x.status().ToString() << std::endl;
    return 1;
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> y = 
      client->BufferFromHostLiteral(y_literal, *device_memory_space);
  if (!y.ok()) {
    std::cerr << "Failed to create y buffer: " 
              << y.status().ToString() << std::endl;
    return 1;
  }

  // 执行计算
  std::cout << "Executing computation on GPU..." << std::endl;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> result = 
      executable->Execute({{alpha->get(), x->get(), y->get()}}, {});
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

int execute_gemm_program(std::unique_ptr<PjRtClient> &client,std::unique_ptr<PjRtLoadedExecutable> &executable) { 
  // 构造输入参数
  std::cout << "Creating input parameters..." << std::endl;
  
  // 创建第一个输入：32x32的矩阵，值全为1.0
  auto x_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {32, 32});
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      x_literal.Set({i, j}, 1.0f);
    }
  }
  
  // 创建第二个输入：32x32的矩阵，值全为1.0
  auto y_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {32, 32});
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      y_literal.Set({i, j}, 1.0f);
    }
  }
  
  // 创建第三个输入：3x3x1x8的卷积核，值全为1.0
  auto filters_literal = xla::LiteralUtil::CreateFromDimensions(xla::PrimitiveType::F32, {3, 3, 1, 8});
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 1; ++k) {
        for (int l = 0; l < 8; ++l) {
          filters_literal.Set({i, j, k, l}, 1.0f);
        }
      }
    }
  }

  std::cout << "Computation inputs:" << std::endl;
  std::cout << "\tx: 32x32 matrix with all 1.0s" << std::endl;
  std::cout << "\ty: 32x32 matrix with all 1.0s" << std::endl;
  std::cout << "\tfilters: 3x3x1x8 convolution kernel with all 1.0s" << std::endl;

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

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> y = 
      client->BufferFromHostLiteral(y_literal, *device_memory_space);
  if (!y.ok()) {
    std::cerr << "Failed to create y buffer: " 
              << y.status().ToString() << std::endl;
    return 1;
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> filters = 
      client->BufferFromHostLiteral(filters_literal, *device_memory_space);
  if (!filters.ok()) {
    std::cerr << "Failed to create filters buffer: " 
              << filters.status().ToString() << std::endl;
    return 1;
  }

  // 执行计算
  std::cout << "Executing computation on GPU..." << std::endl;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> result = 
      executable->Execute({{x->get(), y->get(), filters->get()}}, {});
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

absl::StatusOr<std::unique_ptr<xla::HloModule>> convert_stablehlo_to_hlo_module(mlir::OwningOpRef<mlir::ModuleOp> &program) { 
  // 将StableHLO转换为HloModule并打印计算图
  std::cout << "Converting StableHLO to HloModule..." << std::endl;
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_or = 
      xla::ConvertStablehloToHlo(*program);
  if (!hlo_module_or.ok()) {
    std::cerr << "Failed to convert StableHLO to HloModule: " 
              << hlo_module_or.status().ToString() << std::endl;
    return hlo_module_or.status();
  }
  std::unique_ptr<xla::HloModule> hlo_module = std::move(*hlo_module_or);
  
  // 打印HloModule的计算图
  std::cout << "\nHloModule computation graph:" << std::endl;
  std::cout << hlo_module->ToString() << std::endl;
  return std::move(hlo_module);
}

int compile_by_stablehlo(mlir::OwningOpRef<mlir::ModuleOp> &program,std::unique_ptr<PjRtClient> &client){
  // 编译StableHLO程序
  std::cout << "Compiling StableHLO program for GPU..." << std::endl;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable_or = 
      client->CompileAndLoad(*program, CompileOptions{});
  if (!executable_or.ok()) {
    std::cerr << "Failed to compile StableHLO program: " 
              << executable_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtLoadedExecutable> executable = std::move(*executable_or);

  std::cout << "Successfully compiled StableHLO program for GPU!" << std::endl;
  std::cout << "Executable created with " << executable->addressable_devices().size() 
            << " devices." << std::endl;
  return 0;
}

// 主函数
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <stablehlo_file.mlir>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  
  // 加载StableHLO程序
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> program_or = 
      LoadStableHloProgram(program_path);
  if (!program_or.ok()) {
    std::cerr << "Failed to load StableHLO program: " 
              << program_or.status().ToString() << std::endl;
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> program = std::move(*program_or);

  // 创建GPU客户端:xla/pjrt/gpu/se_gpu_pjrt_client.cc
  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or = CreateGpuClient();
  if (!client_or.ok()) {
    std::cerr << "Failed to create GPU client: " 
              << client_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtClient> client = std::move(*client_or);

  //compile_by_stablehlo(program,client);

  // 转换StableHLO到HloModule
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_or = 
      convert_stablehlo_to_hlo_module(program);
  if (!hlo_module_or.ok()) {
    std::cerr << "Failed to convert StableHLO to HloModule: " 
              << hlo_module_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<xla::HloModule> hlo_module = std::move(*hlo_module_or);

  // 使用HloModule进行编译
  std::cout << "Compiling HloModule for GPU..." << std::endl;
  XlaComputation computation(hlo_module->ToProto());
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable_or = 
      client->CompileAndLoad(computation, CompileOptions{});
  if (!executable_or.ok()) {
    std::cerr << "Failed to compile HloModule: " 
              << executable_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtLoadedExecutable> executable = std::move(*executable_or);

  std::cout << "Successfully compiled StableHLO program for GPU!" << std::endl;
  std::cout << "Executable created with " << executable->addressable_devices().size() 
            << " devices." << std::endl;

  execute_gemm_program(client, executable);
  return 0;
}

}  // namespace xla

int main(int argc, char** argv) {
  return xla::main(argc, argv);
}
/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

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

  std::cout << "Loaded StableHLO program from " << program_path << ":\n"
            << program_string << std::endl;

  return mlir::parseSourceString<mlir::ModuleOp>(program_string, context);
}

// 创建CPU PJRT客户端
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateCpuClient() {
  xla::CpuClientOptions options;
  options.cpu_device_count = 4;
  return xla::GetXlaPjrtCpuClient(options);
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

  // 创建CPU客户端
  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or = CreateCpuClient();
  if (!client_or.ok()) {
    std::cerr << "Failed to create CPU client: " 
              << client_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtClient> client = std::move(*client_or);

  // 编译StableHLO程序
  std::cout << "Compiling StableHLO program..." << std::endl;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable_or = 
      client->CompileAndLoad(*program, CompileOptions{});
  if (!executable_or.ok()) {
    std::cerr << "Failed to compile StableHLO program: " 
              << executable_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<PjRtLoadedExecutable> executable = std::move(*executable_or);

  std::cout << "Successfully compiled StableHLO program!" << std::endl;
  std::cout << "Executable created with " << executable->addressable_devices().size() 
            << " devices." << std::endl;

  
  return 0;
}

}  // namespace xla

int main(int argc, char** argv) {
  return xla::main(argc, argv);
}
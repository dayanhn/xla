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
#include "xla/examples/axpy/compiler_utils.h"

namespace xla {

// 主函数
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlomodule_file.mlir>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  // 加载HloModule
  std::unique_ptr<xla::HloModule> hlo_module = xla::Compiler_LoadHloModuleFromFile(program_path);
  if(!hlo_module){ return 1;}

  // 创建GPU客户端
  std::unique_ptr<PjRtClient> client = xla::Compiler_CreateGpuClient();
  if(!client){ return 1;}

  // 获取Backend
  xla::Backend* backend = nullptr;
  se::StreamExecutor* executor = nullptr;

  xla::Compiler::CompileOptions compile_options;
  if(xla::Compiler_GetGpuBackend(client, backend, executor, compile_options.slice_size) != 0){
    return 1;
  }

  std::cout << "Running Backend..." << std::endl;
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
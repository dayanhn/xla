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

#ifndef XLA_EXAMPLES_AXPY_COMPILER_UTILS_H_
#define XLA_EXAMPLES_AXPY_COMPILER_UTILS_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// 从文件读取HloModule内容
std::unique_ptr<HloModule> Compiler_LoadHloModuleFromFile(
    absl::string_view file_path);

// 创建GPU PJRT客户端
std::unique_ptr<PjRtClient> Compiler_CreateGpuClient();

// 获取GPU后端信息
int Compiler_GetGpuBackend(std::unique_ptr<PjRtClient> &client, 
                  xla::Backend* &backend, 
                  se::StreamExecutor* &executor, 
                  int64_t &slice_size);

// 读取StableHLO文件并解析为MLIR模块
mlir::OwningOpRef<mlir::ModuleOp> Compiler_LoadStableHloProgram(
    absl::string_view program_path);

// 将StableHLO转换为HloModule
std::unique_ptr<xla::HloModule> Compiler_GetHloModule(
    mlir::OwningOpRef<mlir::ModuleOp> &program);

int Compiler_GetGpuDeviceInfo(std::unique_ptr<PjRtClient> &client, se::DeviceDescription &device_info);
}  // namespace xla

#endif  // XLA_EXAMPLES_AXPY_COMPILER_UTILS_H_

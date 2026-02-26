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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/shape_util.h"  // for ShapeUtil::ByteSizeOfElements
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {

// 从文件读取HloModule内容并解析
absl::StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
    absl::string_view hlo_text) {
  HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);
  
  // 创建VerifiedHloModule
  auto module = std::make_unique<VerifiedHloModule>(
      "test_module", config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision=*/false, ShapeUtil::ByteSizeOfElements);
  
  // 解析HloModule内容
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  
  return module;
}

// 从文件读取HloModule内容
absl::StatusOr<std::unique_ptr<VerifiedHloModule>> LoadHloModuleFromFile(
    absl::string_view file_path) {
  // 读取文件内容到字符串
  std::string hlo_text;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(file_path), &hlo_text));

  std::cout << "Loaded HloModule from " << file_path << " successfully.\n";

  // 解析HloModule
  return ParseAndReturnVerifiedModule(hlo_text);
}

// 运行PriorityFusion Pass并打印结果
absl::Status RunPriorityFusionAnalysis(std::unique_ptr<VerifiedHloModule> verified_module) {
  // 获取HloModule
  HloModule* hlo_module = verified_module.get();
  
  // 打印原始HloModule
  std::cout << "\nOriginal HloModule:\n";
  std::cout << hlo_module->ToString() << std::endl;

  // 创建PriorityFusion对象
  se::DeviceDescription device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  AliasInfo alias_info;
  GpuHloCostAnalysis::Options cost_analysis_options;
  cost_analysis_options.count_multiple_input_accesses = true;
  
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, device_info, &alias_info,
      cost_analysis_options, &mlir_context);

  // 运行PriorityFusion Pass
  std::cout << "\nRunning PriorityFusion Pass..." << std::endl;
  absl::StatusOr<bool> result = priority_fusion.Run(hlo_module);
  if (!result.ok()) {
    return result.status();
  }
  
  bool changed = *result;
  std::cout << "PriorityFusion Pass completed. Changed: " << (changed ? "yes" : "no") << std::endl;

  // 打印处理后的HloModule
  std::cout << "\nProcessed HloModule:\n";
  std::cout << hlo_module->ToString() << std::endl;

  // 分析融合结果
  std::cout << "\nFusion analysis:" << std::endl;
  int fusion_count = 0;
  for (auto computation : hlo_module->computations()) {
    if (computation->FusionInstruction()) {
      fusion_count++;
      HloInstruction* fusion_instr = computation->FusionInstruction();
      std::cout << "Fusion " << fusion_count << ": " << fusion_instr->name() << " (kind: " << 
          (fusion_instr->fusion_kind() == HloInstruction::FusionKind::kLoop ? "kLoop" : 
           fusion_instr->fusion_kind() == HloInstruction::FusionKind::kInput ? "kInput" : 
           fusion_instr->fusion_kind() == HloInstruction::FusionKind::kCustom ? "kCustom" : "unknown") << ")" << std::endl;
      
      // 检查是否是Triton融合
      if (IsGenericTritonFusion(*fusion_instr)) {
        std::cout << "  - This is a Triton fusion" << std::endl;
      }
    }
  }
  std::cout << "Total fusions: " << fusion_count << std::endl;

  return absl::OkStatus();
}

}  // namespace gpu

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlo_module_file>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  
  // 加载HloModule
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_or = 
      gpu::LoadHloModuleFromFile(program_path);
  if (!module_or.ok()) {
    std::cerr << "Failed to load HloModule: " 
              << module_or.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<VerifiedHloModule> module = std::move(*module_or);
  
   // 打印HloModule的计算图
  std::cout << "\nHloModule computation graph:" << std::endl;
  std::cout << module->ToString() << std::endl;



  // 运行PriorityFusion分析
  absl::Status status = gpu::RunPriorityFusionAnalysis(std::move(module));
  if (!status.ok()) {
    std::cerr << "Failed to run PriorityFusion analysis: " 
              << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "\nPriorityFusion analysis completed successfully!" << std::endl;
  return 0;
}

}  // namespace xla

int main(int argc, char** argv) {
  return xla::main(argc, argv);
}
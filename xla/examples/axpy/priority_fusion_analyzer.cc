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
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/debug_options_flags.h"
#include "xla/service/dump.h"
#include "xla/service/dump_options.h"
#include "xla/examples/axpy/compiler_utils.h"

namespace xla {
namespace gpu {

// 运行PriorityFusion Pass并打印结果
absl::Status RunPriorityFusionAnalysis(std::unique_ptr<HloModule> hlo_module) {
  
  // 打印原始HloModule
  DumpHloModuleDuringPassIfEnabled("priority_fusion", "before", *hlo_module);

  // 创建GPU客户端以获取真实设备信息
  std::unique_ptr<PjRtClient> client = xla::Compiler_CreateGpuClient();
  if(!client){ return absl::InternalError("Failed to create GPU client");}

  se::DeviceDescription device_info;
  if(xla::Compiler_GetGpuDeviceInfo(client, device_info) != 0){
    return absl::InternalError("Failed to get GPU device info");
  }

  // 创建PriorityFusion对象
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  AliasInfo alias_info;
  GpuHloCostAnalysis::Options cost_analysis_options;
  cost_analysis_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion( /*thread_pool=*/nullptr, device_info, &alias_info,cost_analysis_options, &mlir_context);

  // 运行PriorityFusion Pass
  std::cout << "\nRunning PriorityFusion Pass..." << std::endl;
  absl::StatusOr<bool> result = priority_fusion.Run(hlo_module.get());
  if (!result.ok()) {
    return result.status();
  }
  
  bool changed = *result;
  std::cout << "PriorityFusion Pass completed. Changed: " << (changed ? "yes" : "no") << std::endl;

  // 打印处理后的HloModule
  DumpHloModuleDuringPassIfEnabled("priority_fusion", "after", *hlo_module);

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
}  // namespace xla

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlo_module_file>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  std::unique_ptr<xla::HloModule> module = xla::Compiler_LoadHloModuleFromFile(program_path);
  if(!module){ return 1;}

  // 运行PriorityFusion分析
  absl::Status status = xla::gpu::RunPriorityFusionAnalysis(std::move(module));
  if (!status.ok()) {
    std::cerr << "Failed to run PriorityFusion analysis: " 
              << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "\nPriorityFusion analysis completed successfully!" << std::endl;
  return 0;
}



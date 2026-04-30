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
#include "xla/backends/ascend/transforms/aclnn_fusion_pass.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ascend {

std::unique_ptr<HloModule> LoadHloModuleFromFile(absl::string_view file_path) {
  std::string hlo_text;
  absl::Status status = tsl::ReadFileToString(
      tsl::Env::Default(), std::string(file_path), &hlo_text);
  if (!status.ok()) {
    std::cerr << "Failed to read HloModule from " << file_path 
              << ": " << status.ToString() << std::endl;
    return nullptr;
  }

  std::cout << "Loaded HloModule from " << file_path << " successfully.\n";

  HloModuleConfig config;
  config.set_replica_count(1);
  config.set_num_partitions(1);

  absl::StatusOr<std::unique_ptr<HloModule>> module_or = 
      ParseAndReturnUnverifiedModule(hlo_text, config);
  if (!module_or.ok()) {
    std::cerr << "Failed to load HloModule: " 
              << module_or.status().ToString() << std::endl;
    return nullptr;
  }

  return std::move(*module_or);
}

absl::Status RunAclnnFusionAnalysis(std::unique_ptr<HloModule> hlo_module) {
  //std::cout << "\n=== Before ACLNN Fusion Pass ===" << std::endl;
  //std::cout << hlo_module->ToString() << std::endl;

  stream_executor::GpuTargetConfigProto proto;
  absl::StatusOr<xla::gpu::GpuTargetConfig> target_config_or = 
      xla::gpu::GpuTargetConfig::FromProto(proto);
  if (!target_config_or.ok()) {
    return target_config_or.status();
  }
  xla::gpu::GpuTargetConfig gpu_target_config = *target_config_or;

  std::cout << "\nRunning ACLNN Fusion Pass..." << std::endl;
  absl::Status status = RunAclnnFusionPass(hlo_module.get(), gpu_target_config);
  if (!status.ok()) {
    return status;
  }

  std::cout << "ACLNN Fusion Pass completed successfully." << std::endl;

  std::cout << "\n=== After ACLNN Fusion Pass ===" << std::endl;
  std::cout << hlo_module->ToString() << std::endl;

  return absl::OkStatus();
}

}  // namespace ascend
}  // namespace xla

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <hlo_module_file>" << std::endl;
    return 1;
  }

  std::string program_path = argv[1];
  std::unique_ptr<xla::HloModule> module = xla::ascend::LoadHloModuleFromFile(program_path);
  if (!module) {
    return 1;
  }

  absl::Status status = xla::ascend::RunAclnnFusionAnalysis(std::move(module));
  if (!status.ok()) {
    std::cerr << "Failed to run ACLNN fusion analysis: " 
              << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "\nACLNN fusion analysis completed successfully!" << std::endl;
  return 0;
}
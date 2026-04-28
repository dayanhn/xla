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

#include "xla/backends/ascend/transforms/aclnn_config.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xla/backends/ascend/transforms/aclnn_targets.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ascend {

std::string AclnnGemmConfig::ToString() const {
  return absl::StrCat(
      "alpha=", alpha,
      ",beta=", beta,
      ",transpose_a=", transpose_a,
      ",transpose_b=", transpose_b,
      ",lhs_stride=", lhs_stride,
      ",rhs_stride=", rhs_stride,
      ",has_bias=", has_bias);
}

absl::Status AclnnGemmConfig::FromString(const std::string& config_str) {
  if (config_str.empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> parts = absl::StrSplit(config_str, ',');
  for (const auto& part : parts) {
    std::vector<std::string> key_value = absl::StrSplit(part, '=');
    if (key_value.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid config part: ", part));
    }
    const std::string& key = key_value[0];
    const std::string& value = key_value[1];

    if (key == "alpha") {
      alpha = std::stof(value);
    } else if (key == "beta") {
      beta = std::stof(value);
    } else if (key == "transpose_a") {
      transpose_a = std::stoi(value);
    } else if (key == "transpose_b") {
      transpose_b = std::stoi(value);
    } else if (key == "lhs_stride") {
      lhs_stride = std::stoi(value);
    } else if (key == "rhs_stride") {
      rhs_stride = std::stoi(value);
    } else if (key == "has_bias") {
      has_bias = (value == "1");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AclnnConfig>> AclnnConfigFactory::CreateConfig(
    const std::string& target) {
  if (target == kAclnnGemmCallTarget) {
    return std::make_unique<AclnnGemmConfig>();
  }
  // Add other targets here as needed
  return absl::InvalidArgumentError(absl::StrCat("Unknown ACLNN target: ", target));
}

absl::StatusOr<std::unique_ptr<AclnnConfig>> ParseAclnnConfig(
    absl::string_view target, absl::string_view config_str) {
  auto config_or = AclnnConfigFactory::CreateConfig(std::string(target));
  if (!config_or.ok()) {
    return config_or.status();
  }
  std::unique_ptr<AclnnConfig> config = std::move(config_or).value();
  TF_RETURN_IF_ERROR(config->FromString(std::string(config_str)));
  return config;
}

std::string SerializeAclnnConfig(const AclnnConfig& config) {
  return config.ToString();
}

}  // namespace ascend
}  // namespace xla

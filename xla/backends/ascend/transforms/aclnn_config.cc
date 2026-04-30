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

std::string AclnnConvolutionConfig::ToString() const {
  std::string stride_str, padding_str, dilation_str, output_padding_str;
  for (size_t i = 0; i < stride.size(); ++i) {
    if (i > 0) stride_str += ",";
    stride_str += absl::StrCat(stride[i]);
  }
  for (size_t i = 0; i < padding.size(); ++i) {
    if (i > 0) padding_str += ",";
    padding_str += absl::StrCat(padding[i]);
  }
  for (size_t i = 0; i < dilation.size(); ++i) {
    if (i > 0) dilation_str += ",";
    dilation_str += absl::StrCat(dilation[i]);
  }
  for (size_t i = 0; i < output_padding.size(); ++i) {
    if (i > 0) output_padding_str += ",";
    output_padding_str += absl::StrCat(output_padding[i]);
  }

  return absl::StrCat(
      "stride=", stride_str,
      ",padding=", padding_str,
      ",dilation=", dilation_str,
      ",transposed=", (transposed ? "1" : "0"),
      ",output_padding=", output_padding_str,
      ",groups=", groups,
      ",cube_math_type=", static_cast<int>(cube_math_type),
      ",has_bias=", (has_bias ? "1" : "0"));
}

absl::Status AclnnConvolutionConfig::FromString(const std::string& config_str) {
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

    if (key == "stride") {
      stride.clear();
      for (const auto& s : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(s, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid stride value: ", s));
        }
        stride.push_back(val);
      }
    } else if (key == "padding") {
      padding.clear();
      for (const auto& p : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(p, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid padding value: ", p));
        }
        padding.push_back(val);
      }
    } else if (key == "dilation") {
      dilation.clear();
      for (const auto& d : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(d, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid dilation value: ", d));
        }
        dilation.push_back(val);
      }
    } else if (key == "transposed") {
      transposed = (value == "1");
    } else if (key == "output_padding") {
      output_padding.clear();
      for (const auto& o : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(o, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid output_padding value: ", o));
        }
        output_padding.push_back(val);
      }
    } else if (key == "groups") {
      if (!absl::SimpleAtoi(value, &groups)) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid groups value: ", value));
      }
    } else if (key == "cube_math_type") {
      int cube_val;
      if (!absl::SimpleAtoi(value, &cube_val)) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid cube_math_type value: ", value));
      }
      cube_math_type = static_cast<int8_t>(cube_val);
    } else if (key == "has_bias") {
      has_bias = (value == "1");
    }
  }
  return absl::OkStatus();
}

std::string AclnnConvolutionBackwardConfig::ToString() const {
  std::string stride_str, padding_str, dilation_str, output_padding_str, output_mask_str;
  for (size_t i = 0; i < stride.size(); ++i) {
    if (i > 0) stride_str += ",";
    stride_str += absl::StrCat(stride[i]);
  }
  for (size_t i = 0; i < padding.size(); ++i) {
    if (i > 0) padding_str += ",";
    padding_str += absl::StrCat(padding[i]);
  }
  for (size_t i = 0; i < dilation.size(); ++i) {
    if (i > 0) dilation_str += ",";
    dilation_str += absl::StrCat(dilation[i]);
  }
  for (size_t i = 0; i < output_padding.size(); ++i) {
    if (i > 0) output_padding_str += ",";
    output_padding_str += absl::StrCat(output_padding[i]);
  }
  for (size_t i = 0; i < output_mask.size(); ++i) {
    if (i > 0) output_mask_str += ",";
    output_mask_str += output_mask[i] ? "1" : "0";
  }

  return absl::StrCat(
      "stride=", stride_str,
      ",padding=", padding_str,
      ",dilation=", dilation_str,
      ",transposed=", (transposed ? "1" : "0"),
      ",output_padding=", output_padding_str,
      ",groups=", groups,
      ",cube_math_type=", static_cast<int>(cube_math_type),
      ",output_mask=", output_mask_str);
}

absl::Status AclnnConvolutionBackwardConfig::FromString(const std::string& config_str) {
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

    if (key == "stride") {
      stride.clear();
      for (const auto& s : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(s, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid stride value: ", s));
        }
        stride.push_back(val);
      }
    } else if (key == "padding") {
      padding.clear();
      for (const auto& p : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(p, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid padding value: ", p));
        }
        padding.push_back(val);
      }
    } else if (key == "dilation") {
      dilation.clear();
      for (const auto& d : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(d, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid dilation value: ", d));
        }
        dilation.push_back(val);
      }
    } else if (key == "transposed") {
      transposed = (value == "1");
    } else if (key == "output_padding") {
      output_padding.clear();
      for (const auto& o : absl::StrSplit(value, ',')) {
        int64_t val;
        if (!absl::SimpleAtoi(o, &val)) {
          return absl::InvalidArgumentError(absl::StrCat("Invalid output_padding value: ", o));
        }
        output_padding.push_back(val);
      }
    } else if (key == "groups") {
      if (!absl::SimpleAtoi(value, &groups)) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid groups value: ", value));
      }
    } else if (key == "cube_math_type") {
      int cube_val;
      if (!absl::SimpleAtoi(value, &cube_val)) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid cube_math_type value: ", value));
      }
      cube_math_type = static_cast<int8_t>(cube_val);
    } else if (key == "output_mask") {
      output_mask.clear();
      for (const auto& o : absl::StrSplit(value, ',')) {
        output_mask.push_back((o == "1"));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AclnnConfig>> AclnnConfigFactory::CreateConfig(
    const std::string& target) {
  if (target == kAclnnGemmCallTarget) {
    return std::make_unique<AclnnGemmConfig>();
  } else if (target == kAclnnConvolutionCallTarget) {
    return std::make_unique<AclnnConvolutionConfig>();
  } else if (target == kAclnnConvolutionBackwardCallTarget) {
    return std::make_unique<AclnnConvolutionBackwardConfig>();
  }
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

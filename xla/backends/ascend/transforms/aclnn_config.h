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

#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONFIG_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONFIG_H_

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla {
namespace ascend {

// Base class for all ACLNN configs
class AclnnConfig {
 public:
  virtual ~AclnnConfig() = default;
  virtual std::string ToString() const = 0;
  virtual absl::Status FromString(const std::string& config_str) = 0;
};

// ACLNN GEMM config
class AclnnGemmConfig : public AclnnConfig {
 public:
  float alpha = 1.0f;
  float beta = 0.0f;
  int64_t transpose_a = 0;
  int64_t transpose_b = 0;
  int64_t lhs_stride = 0;
  int64_t rhs_stride = 0;
  bool has_bias = false;

  std::string ToString() const override;
  absl::Status FromString(const std::string& config_str) override;
};

// ACLNN Convolution config
class AclnnConvolutionConfig : public AclnnConfig {
 public:
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation = {1, 1};
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups = 1;
  int8_t cube_math_type = 0;
  bool has_bias = false;

  std::string dim_labels;  // e.g., "b01f_01io->b01f" for NHWC input/output

  std::string ToString() const override;
  absl::Status FromString(const std::string& config_str) override;
};

// ACLNN Convolution Backward config
class AclnnConvolutionBackwardConfig : public AclnnConfig {
 public:
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation = {1, 1};
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups = 1;
  int8_t cube_math_type = 0;
  std::vector<bool> output_mask = {true, true, false};  // gradInput, gradWeight, gradBias

  // Dimension labels for layout conversion: input_kernel->output
  // For backward input: output_kernel->input (dimension numbers are swapped)
  // For backward filter: input_output->kernel (dimension numbers are swapped)
  std::string dim_labels;

  //std::vector<int64_t> input_shape;
  //std::vector<int64_t> weight_shape;
  //int64_t input_data_type = 0;
  //int64_t weight_data_type = 0;

  std::string ToString() const override;
  absl::Status FromString(const std::string& config_str) override;
};

// Config factory to create appropriate config based on target
class AclnnConfigFactory {
 public:
  static absl::StatusOr<std::unique_ptr<AclnnConfig>> CreateConfig(
      const std::string& target);
};

// Helper functions for config operations
absl::StatusOr<std::unique_ptr<AclnnConfig>> ParseAclnnConfig(
    absl::string_view target, absl::string_view config_str);

std::string SerializeAclnnConfig(const AclnnConfig& config);

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_CONFIG_H_

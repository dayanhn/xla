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

#include "xla/backends/ascend/runtime/aclnn_thunk.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"

namespace xla {
namespace ascend {

// Create aclTensor from device address and shape
aclTensor* CreateAclTensorFromDeviceAddress(
    const se::DeviceAddress& device_addr, const Shape& shape) {
  // Get ACL data type from XLA primitive type
  aclDataType data_type = ffi::ConvertToAclDataType(shape.element_type());

  // Get shape dimensions
  std::vector<int64_t> dims;
  for (int i = 0; i < shape.rank(); ++i) {
    dims.push_back(shape.dimensions(i));
  }

  // Calculate strides
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }

  // Create aclTensor
  aclTensor* tensor = aclCreateTensor(
      dims.data(),
      dims.size(),
      data_type,
      strides.data(),
      0,
      ACL_FORMAT_ND,
      dims.data(),
      dims.size(),
      const_cast<void*>(device_addr.opaque()));
  if (tensor == nullptr) {
    LOG(FATAL) << "Failed to create aclTensor: " << aclGetRecentErrMsg();
    return nullptr;
  }

  return tensor;
}

AclnnThunk::AclnnThunk(gpu::Thunk::ThunkInfo thunk_info, std::string op_name,
                       std::vector<gpu::Thunk::NullableShapedSlice> operands,
                       std::vector<gpu::Thunk::NullableShapedSlice> results,
                       std::vector<Param> params)
    : gpu::Thunk(gpu::Thunk::kCustomCall, thunk_info),
      op_name_(std::move(op_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      params_(std::move(params)) {
}

absl::Status AclnnThunk::ExecuteOnStream(const gpu::ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream* stream,
                      GetStreamForExecution(execution_stream_id(), params));

  // Prepare runtime parameters
  std::vector<Param> runtime_params = params_;

  // Replace tensor placeholders with actual aclTensor objects
  size_t tensor_index = 0;
  for (size_t i = 0; i < operands_.size(); ++i) {
    if (operands_[i].has_value()) {
      const auto& slice = operands_[i].value();
      se::DeviceAddress device_addr = params.buffer_allocations->GetDeviceAddress(slice.slice);
      aclTensor* tensor = CreateAclTensorFromDeviceAddress(device_addr, slice.shape);
      runtime_params[tensor_index++] = tensor;
    }
  }

  for (size_t i = 0; i < results_.size(); ++i) {
    if (results_[i].has_value()) {
      const auto& slice = results_[i].value();
      se::DeviceAddress device_addr = params.buffer_allocations->GetDeviceAddress(slice.slice);
      aclTensor* tensor = CreateAclTensorFromDeviceAddress(device_addr, slice.shape);
      runtime_params[tensor_index++] = tensor;
    }
  }

  // Call the aclnn operation using the macro
  if (op_name_ == "aclnnCast") {
    // For aclnnCast, parameters should be: input_tensor, output_tensor
    auto input_tensor = std::get<aclTensor*>(runtime_params[0]);
    auto output_tensor = std::get<aclTensor*>(runtime_params[1]);
    EXEC_ACLNN_CMD(aclnnCast, input_tensor, output_tensor);
  } else if (op_name_ == "aclnnMuls") {
    // For aclnnMuls, parameters should be: input_tensor, other_scalar, output_tensor
    auto input_tensor = std::get<aclTensor*>(runtime_params[0]);
    auto other = std::get<float>(runtime_params[1]);
    auto output_tensor = std::get<aclTensor*>(runtime_params[2]);
    // Create scalar
    aclScalar* other_scalar = nullptr;
    aclError ret = aclCreateScalar(&other_scalar, &other, ACL_FLOAT);
    if (ret != ACL_SUCCESS) {
      return absl::InternalError("Failed to create scalar: " + std::string(aclGetRecentErrMsg()));
    }
    EXEC_ACLNN_CMD(aclnnMuls, input_tensor, other_scalar, output_tensor);
    aclDestroyScalar(other_scalar);
  } else if (op_name_ == "aclnnMaxDim") {
    // For aclnnMaxDim, parameters should be: input_tensor, dim, keepdim, output_tensor, indices_tensor
    auto input_tensor = std::get<aclTensor*>(runtime_params[0]);
    auto dim = std::get<int64_t>(runtime_params[1]);
    auto keepdim = std::get<bool>(runtime_params[2]);
    auto output_tensor = std::get<aclTensor*>(runtime_params[3]);
    auto indices_tensor = std::get<aclTensor*>(runtime_params[4]);
    EXEC_ACLNN_CMD(aclnnMaxDim, input_tensor, dim, keepdim, output_tensor, indices_tensor);
  } else if (op_name_ == "aclnnGemm") {
    // For aclnnGemm, parameters should be: A, B, C, alpha, beta, transA, transB, out
    auto A = std::get<aclTensor*>(runtime_params[0]);
    auto B = std::get<aclTensor*>(runtime_params[1]);
    auto C = std::get<aclTensor*>(runtime_params[2]);
    auto alpha = std::get<float>(runtime_params[3]);
    auto beta = std::get<float>(runtime_params[4]);
    auto transA = std::get<int64_t>(runtime_params[5]);
    auto transB = std::get<int64_t>(runtime_params[6]);
    auto out = std::get<aclTensor*>(runtime_params[7]);
    EXEC_ACLNN_CMD(aclnnGemm, A, B, C, alpha, beta, transA, transB, out);
  } else {
    return absl::InternalError("Unsupported aclnn operation: " + op_name_);
  }

  // Destroy created tensors
  for (auto& param : runtime_params) {
    if (std::holds_alternative<aclTensor*>(param)) {
      aclDestroyTensor(std::get<aclTensor*>(param));
    }
  }

  return absl::OkStatus();
}

}  // namespace ascend
}  // namespace xla

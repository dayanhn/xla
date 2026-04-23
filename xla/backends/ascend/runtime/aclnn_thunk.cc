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
#include "xla/backends/ascend/runtime/aclnn_api_util.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/service/buffer_assignment.h"
#include "third_party/acl/inc/acl/acl.h"

namespace xla {
namespace ascend {

AclnnThunk::AclnnThunk(gpu::Thunk::ThunkInfo thunk_info, std::string op_name,
                       std::vector<NullableShapedSlice> operands,
                       std::vector<NullableShapedSlice> results,
                       std::vector<Param> params)
    : gpu::Thunk(gpu::Thunk::kCustomCall, thunk_info),
      op_name_(std::move(op_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      params_(std::move(params)) {
}

absl::Status AclnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream* stream,
                      GetStreamForExecution(execution_stream_id(), params));

  // Call the aclnn operation using the macro with direct parameters
  if (op_name_ == "aclnnCast") {
    // For aclnnCast, parameters should be: input_tensor, output_tensor
    CHECK(operands_.size() == 1 && results_.size() == 1) << "aclnnCast requires 1 input and 1 output";
    const auto& input_slice = operands_[0].value();
    const auto& output_slice = results_[0].value();
    aclTensor* input_tensor = ConvertType(*params.buffer_allocations, input_slice.slice, input_slice.shape);
    aclTensor* output_tensor = ConvertType(*params.buffer_allocations, output_slice.slice, output_slice.shape);
    EXEC_ACLNN_CMD(aclnnCast, stream, input_tensor, output_tensor);
  } else if (op_name_ == "aclnnMuls") {
    // For aclnnMuls, parameters should be: input_tensor, other_scalar, output_tensor
    CHECK(operands_.size() == 1 && results_.size() == 1 && params_.size() == 1) << "aclnnMuls requires 1 input, 1 output, and 1 scalar parameter";
    const auto& input_slice = operands_[0].value();
    const auto& output_slice = results_[0].value();
    auto other = std::get<float>(params_[0]);
    aclTensor* input_tensor = ConvertType(*params.buffer_allocations, input_slice.slice, input_slice.shape);
    aclTensor* output_tensor = ConvertType(*params.buffer_allocations, output_slice.slice, output_slice.shape);
    EXEC_ACLNN_CMD(aclnnMuls, stream, input_tensor, other, PrimitiveType::F32, output_tensor);
  } else if (op_name_ == "aclnnMaxDim") {
    // For aclnnMaxDim, parameters should be: input_tensor, dim, keepdim, output_tensor, indices_tensor
    CHECK(operands_.size() == 1 && results_.size() == 2 && params_.size() == 2) << "aclnnMaxDim requires 1 input, 2 outputs, and 2 parameters (dim, keepdim)";
    const auto& input_slice = operands_[0].value();
    const auto& output_slice = results_[0].value();
    const auto& indices_slice = results_[1].value();
    auto dim = std::get<int64_t>(params_[0]);
    auto keepdim = std::get<bool>(params_[1]);
    aclTensor* input_tensor = ConvertType(*params.buffer_allocations, input_slice.slice, input_slice.shape);
    aclTensor* output_tensor = ConvertType(*params.buffer_allocations, output_slice.slice, output_slice.shape);
    aclTensor* indices_tensor = ConvertType(*params.buffer_allocations, indices_slice.slice, indices_slice.shape);
    EXEC_ACLNN_CMD(aclnnMaxDim, stream, input_tensor, dim, keepdim, output_tensor, indices_tensor);
  } else if (op_name_ == "aclnnGemm") {
    // For aclnnGemm, parameters should be: A, B, C, alpha, beta, transA, transB, out, cubeMathType
    CHECK(operands_.size() == 3 && results_.size() >= 1 && params_.size() == 4) << "aclnnGemm requires 3 inputs, 1 output, and 4 parameters (alpha, beta, transA, transB)";
    const auto& A_slice = operands_[0].value();
    const auto& B_slice = operands_[1].value();
    const auto& C_slice = operands_[2].value();
    const auto& out_slice = results_[0].value();
    auto alpha = std::get<float>(params_[0]);
    auto beta = std::get<float>(params_[1]);
    auto transA = std::get<int64_t>(params_[2]);
    auto transB = std::get<int64_t>(params_[3]);
    int8_t cubeMathType = 0;  // Default value for cubeMathType
    aclTensor* A_tensor = ConvertType(*params.buffer_allocations, A_slice.slice, A_slice.shape);
    aclTensor* B_tensor = ConvertType(*params.buffer_allocations, B_slice.slice, B_slice.shape);
    aclTensor* C_tensor = ConvertType(*params.buffer_allocations, C_slice.slice, C_slice.shape);
    aclTensor* out_tensor = ConvertType(*params.buffer_allocations, out_slice.slice, out_slice.shape);
    EXEC_ACLNN_CMD(aclnnGemm, stream, A_tensor, B_tensor, C_tensor, alpha, beta, transA, transB, out_tensor, cubeMathType);
  } else {
    return absl::InternalError("Unsupported aclnn operation: " + op_name_);
  }

  return absl::OkStatus();
}

}  // namespace ascend
}  // namespace xla

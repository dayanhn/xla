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

  // Helper lambda to create TensorTriplet from NullableShapedSlice
  auto make_triplet = [&](const NullableShapedSlice& slice) -> TensorTriplet {
    return TensorTriplet{
      params.buffer_allocations,
      slice.value().slice,
      slice.value().shape
    };
  };

  // Call the aclnn operation using the macro with TensorTriplet objects
  if (op_name_ == "aclnnCast") {
    CHECK(operands_.size() == 1 && results_.size() == 1) << "aclnnCast requires 1 input and 1 output";
    EXEC_ACLNN_CMD(aclnnCast, stream, make_triplet(operands_[0]), make_triplet(results_[0]));
  } else if (op_name_ == "aclnnMuls") {
    CHECK(operands_.size() == 1 && results_.size() == 1 && params_.size() == 1) << "aclnnMuls requires 1 input, 1 output, and 1 scalar parameter";
    auto other = std::get<float>(params_[0]);
    EXEC_ACLNN_CMD(aclnnMuls, stream, make_triplet(operands_[0]), other, PrimitiveType::F32, make_triplet(results_[0]));
  } else if (op_name_ == "aclnnMaxDim") {
    CHECK(operands_.size() == 1 && results_.size() == 2 && params_.size() == 2) << "aclnnMaxDim requires 1 input, 2 outputs, and 2 parameters (dim, keepdim)";
    auto dim = std::get<int64_t>(params_[0]);
    auto keepdim = std::get<bool>(params_[1]);
    EXEC_ACLNN_CMD(aclnnMaxDim, stream, make_triplet(operands_[0]), dim, keepdim, make_triplet(results_[0]), make_triplet(results_[1]));
  } else if (op_name_ == "aclnnGemm") {
    CHECK(operands_.size() == 3 && results_.size() >= 1 && params_.size() == 4) << "aclnnGemm requires 3 inputs, 1 output, and 4 parameters (alpha, beta, transA, transB)";
    auto alpha = std::get<float>(params_[0]);
    auto beta = std::get<float>(params_[1]);
    auto transA = std::get<int64_t>(params_[2]);
    auto transB = std::get<int64_t>(params_[3]);
    int8_t cubeMathType = 0;  // Default value for cubeMathType
    EXEC_ACLNN_CMD(aclnnGemm, stream, 
                   make_triplet(operands_[0]), make_triplet(operands_[1]), make_triplet(operands_[2]),
                   alpha, beta, transA, transB, 
                   make_triplet(results_[0]), cubeMathType);
  } else {
    return absl::InternalError("Unsupported aclnn operation: " + op_name_);
  }

  return absl::OkStatus();
}

}  // namespace ascend
}  // namespace xla

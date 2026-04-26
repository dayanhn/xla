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
#include <unordered_map>
#include <functional>

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

// Define a type for the execution function
using ExecuteFunc = std::function<absl::Status(
    const AclnnThunk::ExecuteParams& params,
    se::Stream* stream,
    const std::vector<NullableShapedSlice>& operands,
    const std::vector<NullableShapedSlice>& results,
    const std::vector<AclnnThunk::Param>& params_list,
    const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet)>;

// Create a map of operation names to their execution functions
static const std::unordered_map<std::string, ExecuteFunc> kOpExecutors = {
  {
    "aclnnCast",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1) << "aclnnCast requires 1 input and 1 output";
      // Get the target dtype from the result shape
      auto target_dtype = results[0].value().shape.element_type();
      EXEC_ACLNN_CMD(aclnnCast, stream, make_triplet(operands[0]), target_dtype, make_triplet(results[0]));
      return absl::OkStatus();
    }
  },
  {
    "aclnnTanh",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1) << "aclnnTanh requires 1 input and 1 output";
      EXEC_ACLNN_CMD(aclnnTanh, stream, make_triplet(operands[0]),make_triplet(results[0]));
      return absl::OkStatus();
    }
  },
  {
    "aclnnSqrt",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1) << "aclnnSqrt requires 1 input and 1 output";
      EXEC_ACLNN_CMD(aclnnSqrt, stream, make_triplet(operands[0]), make_triplet(results[0]));
      return absl::OkStatus();
    }
  },
  {
    "aclnnMuls",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 1) << "aclnnMuls requires 1 input, 1 output, and 1 scalar parameter";
      auto other = std::get<float>(params_list[0]);
      EXEC_ACLNN_CMD(aclnnMuls, stream, make_triplet(operands[0]), other, PrimitiveType::F32, make_triplet(results[0]));
      return absl::OkStatus();
    }
  },
  {
    "aclnnMaxDim",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 2 && params_list.size() == 2) << "aclnnMaxDim requires 1 input, 2 outputs, and 2 parameters (dim, keepdim)";
      auto dim = std::get<int64_t>(params_list[0]);
      auto keepdim = std::get<bool>(params_list[1]);
      EXEC_ACLNN_CMD(aclnnMaxDim, stream, make_triplet(operands[0]), dim, keepdim, make_triplet(results[0]), make_triplet(results[1]));
      return absl::OkStatus();
    }
  },
  {
    "aclnnGemm",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 3 && results.size() >= 1 && params_list.size() == 4) << "aclnnGemm requires 3 inputs, 1 output, and 4 parameters (alpha, beta, transA, transB)";
      auto alpha = std::get<float>(params_list[0]);
      auto beta = std::get<float>(params_list[1]);
      auto transA = std::get<int64_t>(params_list[2]);
      auto transB = std::get<int64_t>(params_list[3]);
      int8_t cubeMathType = 0;  // Default value for cubeMathType
      EXEC_ACLNN_CMD(aclnnGemm, stream, 
                     make_triplet(operands[0]), make_triplet(operands[1]), make_triplet(operands[2]),
                     alpha, beta, transA, transB, 
                     make_triplet(results[0]), cubeMathType);
      return absl::OkStatus();
    }
  },
  {
    "aclnnConvolution",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 2 && results.size() == 1 && params_list.size() == 7) << "aclnnConvolution requires 2 inputs (input, weight), 1 output, and 7 parameters (stride, padding, dilation, transposed, outputPadding, groups, cubeMathType)";
      
      // Extract parameters
      auto stride = std::get<std::vector<int64_t>>(params_list[0]);
      auto padding = std::get<std::vector<int64_t>>(params_list[1]);
      auto dilation = std::get<std::vector<int64_t>>(params_list[2]);
      auto transposed = std::get<bool>(params_list[3]);
      auto outputPadding = std::get<std::vector<int64_t>>(params_list[4]);
      auto groups = std::get<int64_t>(params_list[5]);
      auto cubeMathType = std::get<int8_t>(params_list[6]);
      
      // Call aclnnConvolution
      EXEC_ACLNN_CMD(aclnnConvolution, stream,
                     make_triplet(operands[0]),  // input
                     make_triplet(operands[1]),  // weight
                     nullptr,  // bias (none)
                     stride,   // stride
                     padding,  // padding
                     dilation, // dilation
                     transposed, // transposed
                     outputPadding, // outputPadding
                     groups,   // groups
                     make_triplet(results[0]), // output
                     cubeMathType); // cubeMathType

      return absl::OkStatus();
    }
  },
  {
    "aclnnMaxPool",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 6)
          << "aclnnMaxPool requires 1 input, 1 output, and 6 parameters";

      // Extract parameters
      auto kernelShape = std::get<std::vector<int64_t>>(params_list[0]);
      auto strides = std::get<std::vector<int64_t>>(params_list[1]);
      auto autoPad = std::get<int64_t>(params_list[2]);
      auto pads = std::get<std::vector<int64_t>>(params_list[3]);
      auto dilations = std::get<std::vector<int64_t>>(params_list[4]);
      auto ceilMode = std::get<int64_t>(params_list[5]);

      // Call aclnnMaxPool
      EXEC_ACLNN_CMD(aclnnMaxPool, stream,
                     make_triplet(operands[0]),  // self
                     kernelShape,
                     strides,
                     autoPad,
                     pads,
                     dilations,
                     ceilMode,
                     make_triplet(results[0]));  // out

      return absl::OkStatus();
    }
  },
  {
    "aclnnCat",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() >= 2 && results.size() == 1 && params_list.size() == 1)
          << "aclnnCat requires at least 2 inputs, 1 output, and 1 parameter (concat_dim)";

      // Extract the concatenate dimension
      int64_t concat_dim = std::get<int64_t>(params_list[0]);

      // Build tensor list from operands
      std::vector<aclTensor*> tensor_list;
      tensor_list.reserve(operands.size());
      for (const auto& operand : operands) {
        TensorTriplet triplet = make_triplet(operand);
        tensor_list.push_back(triplet.device_memory_data());
      }

      // Call aclnnCat
      EXEC_ACLNN_CMD(aclnnCat, stream,
                     tensor_list,
                     concat_dim,
                     make_triplet(results[0]));

      return absl::OkStatus();
    }
  }
};

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

  // Find the executor for the current operation
  auto it = kOpExecutors.find(op_name_);
  if (it == kOpExecutors.end()) {
    return absl::InternalError("Unsupported aclnn operation: " + op_name_);
  }

  // Execute the operation
  return it->second(params, stream, operands_, results_, params_, make_triplet);
}

}  // namespace ascend
}  // namespace xla

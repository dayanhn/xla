/* Copyright 2018 The JAX Authors

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

#include <memory>
#include <vector>
#include <cstdlib>

#include "xla/backends/ascend/runtime/aclnn_api_util.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/service/buffer_assignment.h"
#include "third_party/acl/inc/acl/acl.h"
#include "aclnnop/aclnn_permute.h"
#include <unordered_map>
#include <functional>

namespace xla {
namespace ascend {

AclnnThunk::AclnnThunk(gpu::Thunk::ThunkInfo thunk_info, std::string op_name,
                       std::vector<NullableShapedSlice> operands,
                       std::vector<NullableShapedSlice> results,
                       std::vector<Param> params,
                       std::vector<aclFormat> operand_formats,
                       std::vector<aclFormat> result_formats)
    : gpu::Thunk(gpu::Thunk::kCustomCall, thunk_info),
      op_name_(std::move(op_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      params_(std::move(params)),
      operand_formats_(std::move(operand_formats)),
      result_formats_(std::move(result_formats)) {
}

// Define a type for the execution function
// make_triplet_with_index takes (index, is_operand) -> TensorTriplet
using ExecuteFunc = std::function<absl::Status(
    const AclnnThunk::ExecuteParams& params,
    se::Stream* stream,
    const std::vector<NullableShapedSlice>& operands,
    const std::vector<NullableShapedSlice>& results,
    const std::vector<AclnnThunk::Param>& params_list,
    const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index)>;

// Create a map of operation names to their execution functions
static const std::unordered_map<std::string, ExecuteFunc> kOpExecutors = {
  {
    "aclnnCast",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1, "aclnnCast requires 1 input and 1 output");
      // Get the target dtype from the result shape
      auto target_dtype = results[0].value().shape.element_type();
      EXEC_ACLNN_CMD(aclnnCast, stream, make_triplet_with_index(0, true), target_dtype, make_triplet_with_index(0, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnTanh",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1, "aclnnTanh requires 1 input and 1 output");
      EXEC_ACLNN_CMD(aclnnTanh, stream, make_triplet_with_index(0, true), make_triplet_with_index(0, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnPermute",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 1, "aclnnPermute requires 1 input, 1 output, and 1 parameter (dimensions)");
      
      // Extract dimensions from variant parameter
      std::vector<int64_t> dims = std::get<std::vector<int64_t>>(params_list[0]);
      
      // Execute permute
      EXEC_ACLNN_CMD(aclnnPermute, stream, make_triplet_with_index(0, true), dims, make_triplet_with_index(0, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnSqrt",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1, "aclnnSqrt requires 1 input and 1 output");
      EXEC_ACLNN_CMD(aclnnSqrt, stream, make_triplet_with_index(0, true), make_triplet_with_index(0, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnMuls",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 1, "aclnnMuls requires 1 input, 1 output, and 1 scalar parameter");
      auto other = std::get<float>(params_list[0]);
      EXEC_ACLNN_CMD(aclnnMuls, stream, make_triplet_with_index(0, true), other, PrimitiveType::F32, make_triplet_with_index(0, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnMaxDim",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream, 
       const std::vector<NullableShapedSlice>& operands, 
       const std::vector<NullableShapedSlice>& results, 
       const std::vector<AclnnThunk::Param>& params_list, 
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 2 && params_list.size() == 2, "aclnnMaxDim requires 1 input, 2 outputs, and 2 parameters (dim, keepdim)");
      auto dim = std::get<int64_t>(params_list[0]);
      auto keepdim = std::get<bool>(params_list[1]);
      EXEC_ACLNN_CMD(aclnnMaxDim, stream, make_triplet_with_index(0, true), dim, keepdim, make_triplet_with_index(0, false), make_triplet_with_index(1, false));
      return absl::OkStatus();
    }
  },
  {
    "aclnnGemm",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 3 && results.size() >= 1 && params_list.size() == 4, "aclnnGemm requires 3 inputs, 1 output, and 4 parameters (alpha, beta, transA, transB)");
      auto alpha = std::get<float>(params_list[0]);
      auto beta = std::get<float>(params_list[1]);
      auto transA = std::get<int64_t>(params_list[2]);
      auto transB = std::get<int64_t>(params_list[3]);
      int8_t cubeMathType = 0;  // Default value for cubeMathType
      EXEC_ACLNN_CMD(aclnnGemm, stream, 
                     make_triplet_with_index(0, true), make_triplet_with_index(1, true), make_triplet_with_index(2, true),
                     alpha, beta, transA, transB, 
                     make_triplet_with_index(0, false), cubeMathType);
      return absl::OkStatus();
    }
  },
  {
    "aclnnConvolution",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK((operands.size() == 2 || operands.size() == 3) && results.size() == 1 && params_list.size() == 7, "aclnnConvolution requires 2 or 3 inputs (input, weight, [bias]), 1 output, and 7 parameters (stride, padding, dilation, transposed, outputPadding, groups, cubeMathType)");
      
      // Extract parameters
      auto stride = std::get<std::vector<int64_t>>(params_list[0]);
      auto padding = std::get<std::vector<int64_t>>(params_list[1]);
      auto dilation = std::get<std::vector<int64_t>>(params_list[2]);
      auto transposed = std::get<bool>(params_list[3]);
      auto outputPadding = std::get<std::vector<int64_t>>(params_list[4]);
      auto groups = std::get<int64_t>(params_list[5]);
      auto cubeMathType = std::get<int8_t>(params_list[6]);
      
      // Prepare optional bias tensor - use aclTensor* directly to avoid type deduction issues
      aclTensor* bias_tensor = nullptr;
      if (operands.size() > 2 && operands[2].has_value()) {
        bias_tensor = ConvertType(make_triplet_with_index(2, true));
      }
      
      // Call aclnnConvolution
      EXEC_ACLNN_CMD(aclnnConvolution, stream,
                     make_triplet_with_index(0, true),  // input
                     make_triplet_with_index(1, true),  // weight
                     bias_tensor,  // bias (optional, aclTensor* type)
                     stride,   // stride
                     padding,  // padding
                     dilation, // dilation
                     transposed, // transposed
                     outputPadding, // outputPadding
                     groups,   // groups
                     make_triplet_with_index(0, false), // output
                     cubeMathType); // cubeMathType

      return absl::OkStatus();
    }
  },
  {
    "aclnnConvolutionBackward",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      auto stride = std::get<std::vector<int64_t>>(params_list[0]);
      auto padding = std::get<std::vector<int64_t>>(params_list[1]);
      auto dilation = std::get<std::vector<int64_t>>(params_list[2]);
      auto transposed = std::get<bool>(params_list[3]);
      auto outputPadding = std::get<std::vector<int64_t>>(params_list[4]);
      auto groups = std::get<int64_t>(params_list[5]);
      auto cubeMathType = std::get<int8_t>(params_list[6]);
      auto outputMask = std::get<std::vector<bool>>(params_list[7]);
      //auto inputShape = std::get<std::vector<int64_t>>(params_list[8]);
      //auto weightShape = std::get<std::vector<int64_t>>(params_list[9]);
      //auto inputDataType = std::get<int64_t>(params_list[10]);
      //auto weightDataType = std::get<int64_t>(params_list[11]);

      int expected_results = (outputMask[0] ? 1 : 0) +
                             (outputMask[1] ? 1 : 0) +
                             (outputMask[2] ? 1 : 0);
      ACLNN_CHECK(params_list.size() == 12 &&
            results.size() == expected_results,
            "aclnnConvolutionBackward: expected " + std::to_string(expected_results) +
            " results, got " + std::to_string(results.size()) +
            ", params_list size=" + std::to_string(params_list.size()));

      aclTensor* grad_output_tensor = ConvertType(make_triplet_with_index(0, true));
      aclTensor* input_tensor = ConvertType(make_triplet_with_index(1, true));
      aclTensor* weight_tensor = ConvertType(make_triplet_with_index(2, true));

      std::vector<int64_t> biasSizes;
      if (outputMask[2]) {
        biasSizes.push_back(MaxShapeDims(operands[2].value().shape));
      }

      int result_idx = 0;
      aclTensor* gradInput_tensor = nullptr;
      if (outputMask[0] && result_idx < results.size()) {
        gradInput_tensor = ConvertType(make_triplet_with_index(result_idx++, false));
      }
      
      aclTensor* gradWeight_tensor = nullptr;
      if (outputMask[1] && result_idx < results.size()) {
        gradWeight_tensor = ConvertType(make_triplet_with_index(result_idx++, false));
      }
      
      aclTensor* gradBias_tensor = nullptr;
      if (outputMask[2] && result_idx < results.size()) {
        gradBias_tensor = ConvertType(make_triplet_with_index(result_idx++, false));
      }

      EXEC_ACLNN_CMD(aclnnConvolutionBackward, stream,
                     grad_output_tensor,
                     input_tensor,
                     weight_tensor,
                     biasSizes,
                     stride,
                     padding,
                     dilation,
                     transposed,
                     outputPadding,
                     groups,
                     outputMask,
                     gradInput_tensor,
                     gradWeight_tensor,
                     gradBias_tensor,
                     cubeMathType);

      return absl::OkStatus();
    }
  },
  {
    "aclnnMaxPool",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 6, "aclnnMaxPool requires 1 input, 1 output, and 6 parameters");

      // Extract parameters
      auto kernelShape = std::get<std::vector<int64_t>>(params_list[0]);
      auto strides = std::get<std::vector<int64_t>>(params_list[1]);
      auto autoPad = std::get<int64_t>(params_list[2]);
      auto pads = std::get<std::vector<int64_t>>(params_list[3]);
      auto dilations = std::get<std::vector<int64_t>>(params_list[4]);
      auto ceilMode = std::get<int64_t>(params_list[5]);

      // Call aclnnMaxPool
      EXEC_ACLNN_CMD(aclnnMaxPool, stream,
                     make_triplet_with_index(0, true),  // self
                     kernelShape,
                     strides,
                     autoPad,
                     pads,
                     dilations,
                     ceilMode,
                     make_triplet_with_index(0, false));  // out

      return absl::OkStatus();
    }
  },
  {
    "aclnnCat",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(size_t, bool)>& make_triplet_with_index) -> absl::Status {
      ACLNN_CHECK(operands.size() >= 2 && results.size() == 1 && params_list.size() == 1, "aclnnCat requires at least 2 inputs, 1 output, and 1 parameter (concat_dim)");

      // Extract the concatenate dimension
      int64_t concat_dim = std::get<int64_t>(params_list[0]);

      // Build tensor list from operands
      std::vector<aclTensor*> tensor_list;
      tensor_list.reserve(operands.size());
      for (size_t i = 0; i < operands.size(); ++i) {
        TensorTriplet triplet = make_triplet_with_index(i, true);
        tensor_list.push_back(ConvertType(triplet));
      }

      // Call aclnnCat
      EXEC_ACLNN_CMD(aclnnCat, stream,
                     tensor_list,
                     concat_dim,
                     make_triplet_with_index(0, false));

      return absl::OkStatus();
    }
  }
};

absl::Status AclnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream* stream,
                      GetStreamForExecution(execution_stream_id(), params));

  // Helper lambda to create TensorTriplet from NullableShapedSlice with format
  auto make_triplet_with_index = [this, &params](size_t index, bool is_operand) -> TensorTriplet {
    const NullableShapedSlice& slice = is_operand ? operands_[index] : results_[index];
    
    // Get the format: use provided format if available, else ACL_FORMAT_ND
    aclFormat format = ACL_FORMAT_ND;
    if (is_operand) {
      if (index < operand_formats_.size()) {
        format = operand_formats_[index];
      }
    } else {
      if (index < result_formats_.size()) {
        format = result_formats_[index];
      }
    }
    
    return TensorTriplet{
      params.buffer_allocations,
      slice.value().slice,
      slice.value().shape,
      format
    };
  };

  // Find the executor for the current operation
  auto it = kOpExecutors.find(op_name_);
  if (it == kOpExecutors.end()) {
    return absl::InternalError("Unsupported aclnn operation: " + op_name_);
  }

  // Execute the operation
  return it->second(params, stream, operands_, results_, params_, make_triplet_with_index);
}

}  // namespace ascend
}  // namespace xla

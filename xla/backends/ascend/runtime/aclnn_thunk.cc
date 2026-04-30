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
#include "aclnnop/aclnn_permute.h"
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
    "aclnnPermute",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      CHECK(operands.size() == 1 && results.size() == 1 && params_list.size() == 1) << "aclnnPermute requires 1 input, 1 output, and 1 parameter (dimensions)";
      
      // Extract dimensions from variant parameter
      std::vector<int64_t> dims = std::get<std::vector<int64_t>>(params_list[0]);
      
      // Execute permute
      EXEC_ACLNN_CMD(aclnnPermute, stream,make_triplet(operands[0]),dims,make_triplet(results[0]));
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
    "aclnnConvolutionBackward",
    [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
       const std::vector<NullableShapedSlice>& operands,
       const std::vector<NullableShapedSlice>& results,
       const std::vector<AclnnThunk::Param>& params_list,
       const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
      // Extract parameters first (used to validate results size)
      auto stride = std::get<std::vector<int64_t>>(params_list[0]);
      auto padding = std::get<std::vector<int64_t>>(params_list[1]);
      auto dilation = std::get<std::vector<int64_t>>(params_list[2]);
      auto transposed = std::get<bool>(params_list[3]);
      auto outputPadding = std::get<std::vector<int64_t>>(params_list[4]);
      auto groups = std::get<int64_t>(params_list[5]);
      auto cubeMathType = std::get<int8_t>(params_list[6]);
      auto outputMask = std::get<std::vector<bool>>(params_list[7]);

      int expected_results = (outputMask[0] ? 1 : 0) +
                             (outputMask[1] ? 1 : 0) +
                             (outputMask[2] ? 1 : 0);
      CHECK(operands.size() >= 2 && operands.size() <= 3 &&
            results.size() == expected_results && params_list.size() == 8)
          << "aclnnConvolutionBackward: expected " << expected_results
          << " results for output_mask=[" << outputMask[0] << ","
          << outputMask[1] << "," << outputMask[2] << "], got "
          << results.size() << " results, " << operands.size() << " operands";

      // Determine output tensor sizes based on outputMask
      // In NCHW weight format: dim 0 = C_out; in HWIO format: last dim = C_out
      // The layout conversion pass ensures NCHW format at this point
      std::vector<int64_t> biasSizes;
      if (outputMask[2]) {
        biasSizes.push_back(operands[1]->shape.dimensions(0));
      }

      // Prepare optional input tensor
      aclTensor* input_tensor = nullptr;
      if (operands.size() > 2) {
        input_tensor = ConvertType(make_triplet(operands[2]));
      }

      // Map results to API parameters compactly based on outputMask.
      // Each true entry in the mask consumes one result, in order:
      //   gradInput (mask[0]), gradWeight (mask[1]), gradBias (mask[2])
      int result_idx = 0;
      aclTensor* gradInput_tensor = nullptr;
      if (outputMask[0] && result_idx < results.size()) {
        gradInput_tensor = ConvertType(make_triplet(results[result_idx++]));
      }
      
      aclTensor* gradWeight_tensor = nullptr;
      if (outputMask[1] && result_idx < results.size()) {
        gradWeight_tensor = ConvertType(make_triplet(results[result_idx++]));
      }
      
      aclTensor* gradBias_tensor = nullptr;
      if (outputMask[2] && result_idx < results.size()) {
        gradBias_tensor = ConvertType(make_triplet(results[result_idx++]));
      }

      // Call aclnnConvolutionBackward
      EXEC_ACLNN_CMD(aclnnConvolutionBackward, stream,
                     make_triplet(operands[0]),  // gradOutput
                     input_tensor,  // input (optional)
                     make_triplet(operands[1]),  // weight
                     biasSizes,  // biasSizes
                     stride,     // stride
                     padding,    // padding
                     dilation,   // dilation
                     transposed, // transposed
                     outputPadding, // outputPadding
                     groups,     // groups
                     outputMask, // outputMask
                     gradInput_tensor,
                     gradWeight_tensor,
                     gradBias_tensor,
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
        tensor_list.push_back(ConvertType(triplet));
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

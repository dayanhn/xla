#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_expand.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/status/status.h"
#include <cstdint>
#include <iostream>
#include <vector>

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Expand operator FFI handler
template <ffi::DataType DType>
ffi::Error ExpandHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, int64_t dim, ffi::ResultBuffer<DType> out) {
  
  // Get self and output shape dimensions
  const auto& self_dims = self.dimensions();
  const auto& out_dims = out->dimensions();
  size_t self_ndim = self_dims.size();
  size_t out_ndim = out_dims.size();
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  
  // Build target size array from output shape
  std::vector<int64_t> size_vec(out_ndim);
  for (size_t i = 0; i < out_ndim; i++) {
    size_vec[i] = out_dims[i];
  }
  aclIntArray* size_array = aclCreateIntArray(size_vec.data(), size_vec.size());
  if (size_array == nullptr) {
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray for size");
  }
  
  aclTensor* self_tensor = nullptr;
  
  // Handle scalar input case
  if (self_ndim == 0) {
    // Scalar input: directly convert to tensor, aclnnExpand will handle broadcast to any shape
    self_tensor = ConvertToAclTensor(self);
    LOG(INFO) << "Handling scalar input for Expand operation";
  } else {
    // Handle non-scalar input case
    // Validate dim parameter: dim should be in valid range [0, out_ndim-1]
    if (dim < 0 || dim >= static_cast<int64_t>(out_ndim)) {
      return ffi::Error::Internal(
          absl::StrCat("Invalid dim: ", dim, ". dim should be in range [0, ", out_ndim - 1, "]"));
    }
    
    // For 1D input with dim pointing to the matching dimension
    // aclnnExpand automatically handles dimension alignment by prepending 1s
    // Example: self=[32], out=[2,32,32,32], dim=1 -> aclnnExpand treats it as [1,32,1,1] -> [2,32,32,32]
    if (self_ndim == 1 && dim == 0 && out_dims[dim] == self_dims[0]) {
      self_tensor = ConvertToAclTensor(self);
      LOG(INFO) << "Handling 1D input broadcast with dim=1 for Expand operation";
    } else {
      // General case: need to reshape self to match output dimensions
      // Build reshaped dimensions: fill with 1, place self dims starting at dim position
      // Example: self=[32], out=[2,32,32,32], dim=1 -> reshape self to [1,32,1,1]
      // Example: self=[10,20], out=[2,10,20,3], dim=2 -> reshape self to [1,10,20,1]
      
      std::vector<int64_t> reshaped_dims(out_ndim, 1);
      for (size_t i = 0; i < self_ndim; i++) {
        reshaped_dims[dim + i] = self_dims[i];
      }
      
      // Create reshaped tensor by constructing a new aclTensor with modified shape
      aclDataType dtype = ConvertToAclDataType<DType>();
      
      // Calculate strides for the reshaped tensor (matching tensor_utils.cc implementation)
      std::vector<int64_t> strides(reshaped_dims.size(), 1);
      for (int i = static_cast<int>(reshaped_dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * reshaped_dims[i + 1];
      }
      
      // Create new tensor with reshaped dimensions but same data
      self_tensor = aclCreateTensor(
          reshaped_dims.data(), 
          reshaped_dims.size(), 
          dtype, 
          strides.data(), 
          0,  // offset
          ACL_FORMAT_ND, 
          reshaped_dims.data(), 
          reshaped_dims.size(),
          const_cast<void*>(self.untyped_data())
      );
      
      if (self_tensor == nullptr) {
        aclDestroyTensor(out_tensor);
        aclDestroyIntArray(size_array);
        return ffi::Error::Internal("Failed to create reshaped aclTensor");
      }
      
      LOG(INFO) << "Reshaped self from [" << absl::StrJoin(self_dims, ",") 
                << "] to [" << absl::StrJoin(reshaped_dims, ",") << "] for broadcast with dim=" << dim;
    }
  }
  
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Expand operation on stream: " << stream;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnExpandGetWorkspaceSize(
      self_tensor, size_array, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(size_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnExpandGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_status != ACL_SUCCESS) {
      aclDestroyTensor(self_tensor);
      aclDestroyTensor(out_tensor);
      aclDestroyIntArray(size_array);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  }

  // Call second stage interface to execute computation
  status = aclnnExpand(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(size_array);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnExpand failed: ", status));
  }
  
  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(size_array);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyIntArray(size_array);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ExpandHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::S64> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::U8>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::U8> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S8>(aclrtStream stream, ffi::Buffer<ffi::DataType::S8> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::S8> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::PRED>(aclrtStream stream, ffi::Buffer<ffi::DataType::PRED> self, int64_t dim, ffi::ResultBuffer<ffi::DataType::PRED> out);

// F32 specialization
ffi::Error ExpandHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, int64_t dim, ffi::ResultBuffer<ffi::F32> out) {
  return ExpandHandlerImpl<ffi::DataType::F32>(stream, self, dim, out);
}

// F16 specialization
ffi::Error ExpandHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, int64_t dim, ffi::ResultBuffer<ffi::F16> out) {
  return ExpandHandlerImpl<ffi::DataType::F16>(stream, self, dim, out);
}

// BF16 specialization
ffi::Error ExpandHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, int64_t dim, ffi::ResultBuffer<ffi::BF16> out) {
  return ExpandHandlerImpl<ffi::DataType::BF16>(stream, self, dim, out);
}

// S32 specialization
ffi::Error ExpandHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, int64_t dim, ffi::ResultBuffer<ffi::S32> out) {
  return ExpandHandlerImpl<ffi::DataType::S32>(stream, self, dim, out);
}

// S64 specialization
ffi::Error ExpandHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, int64_t dim, ffi::ResultBuffer<ffi::S64> out) {
  return ExpandHandlerImpl<ffi::DataType::S64>(stream, self, dim, out);
}

// U8 specialization
ffi::Error ExpandHandlerU8(aclrtStream stream, ffi::Buffer<ffi::U8> self, int64_t dim, ffi::ResultBuffer<ffi::U8> out) {
  return ExpandHandlerImpl<ffi::DataType::U8>(stream, self, dim, out);
}

// S8 specialization
ffi::Error ExpandHandlerS8(aclrtStream stream, ffi::Buffer<ffi::S8> self, int64_t dim, ffi::ResultBuffer<ffi::S8> out) {
  return ExpandHandlerImpl<ffi::DataType::S8>(stream, self, dim, out);
}

// PRED specialization
ffi::Error ExpandHandlerPRED(aclrtStream stream, ffi::Buffer<ffi::PRED> self, int64_t dim, ffi::ResultBuffer<ffi::PRED> out) {
  return ExpandHandlerImpl<ffi::DataType::PRED>(stream, self, dim, out);
}

// Register Expand operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpand,
    ExpandHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandF32,
    ExpandHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandF16,
    ExpandHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandBF16,
    ExpandHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS32,
    ExpandHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS64,
    ExpandHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandU8,
    ExpandHandlerU8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::U8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS8,
    ExpandHandlerS8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandPRED,
    ExpandHandlerPRED,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Attr<int64_t>("dim")
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi

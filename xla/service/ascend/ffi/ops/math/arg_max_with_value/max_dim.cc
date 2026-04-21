#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_max_dim.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of MaxDim operator FFI handler
template <ffi::DataType DType>
ffi::Error MaxDimHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, int64_t dim, bool keepdim, ffi::ResultBuffer<DType> out, ffi::ResultBuffer<ffi::DataType::S32> indices) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  aclTensor* indices_tensor = ConvertToAclTensor(*indices);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for MaxDim operation on stream: " << stream ;
  
  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMaxDimGetWorkspaceSize(
      self_tensor, dim, keepdim, out_tensor, indices_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyTensor(indices_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMaxDimGetWorkspaceSize failed: ", status));
  }
  
  // Allocate internal workspace for ascend platform
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMaxDim(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyTensor(indices_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMaxDim failed: ", status));
  }
  
  aclrtSynchronizeStream(stream);
  
  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyTensor(indices_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error MaxDimHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::DataType::F32> out, ffi::ResultBuffer<ffi::DataType::S32> indices);
template ffi::Error MaxDimHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::DataType::F16> out, ffi::ResultBuffer<ffi::DataType::S32> indices);
template ffi::Error MaxDimHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::DataType::BF16> out, ffi::ResultBuffer<ffi::DataType::S32> indices);

// F32 specialization
ffi::Error MaxDimHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::S32> indices) {
  return MaxDimHandlerImpl<ffi::DataType::F32>(stream, self, dim, keepdim, out, indices);
}

// F16 specialization
ffi::Error MaxDimHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::F16> out, ffi::ResultBuffer<ffi::S32> indices) {
  return MaxDimHandlerImpl<ffi::DataType::F16>(stream, self, dim, keepdim, out, indices);
}

// BF16 specialization
ffi::Error MaxDimHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, int64_t dim, bool keepdim, ffi::ResultBuffer<ffi::BF16> out, ffi::ResultBuffer<ffi::S32> indices) {
  return MaxDimHandlerImpl<ffi::DataType::BF16>(stream, self, dim, keepdim, out, indices);
}

// Register MaxDim operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaxDim,
    MaxDimHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Attr<bool>("keepdim")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaxDimF32,
    MaxDimHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Attr<bool>("keepdim")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaxDimF16,
    MaxDimHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<int64_t>("dim")
        .Attr<bool>("keepdim")
        .Ret<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaxDimBF16,
    MaxDimHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<int64_t>("dim")
        .Attr<bool>("keepdim")
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
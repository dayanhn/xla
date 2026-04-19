#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_matmul.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Matmul operator FFI handler
template <ffi::DataType DType>
ffi::Error MatmulHandlerCublasImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Buffer<DType> mat2, ffi::ResultBuffer<DType> out, ffi::ResultBuffer<ffi::DataType::S8> workspace) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* mat2_tensor = ConvertToAclTensor(mat2);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Matmul operation on stream: " << stream ;

  // Set cubeMathType (default to 0 for now)
  int8_t cubeMathType = 0;
  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMatmulGetWorkspaceSize(
      self_tensor, mat2_tensor, out_tensor, cubeMathType, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(mat2_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMatmulGetWorkspaceSize failed: ", status));
  }
  // Allocate internal workspace for ascend platform
  // Note: The provided workspace parameter is calculated based on cublas,
  // which is not applicable for ascend platform. We need to allocate
  // our own workspace based on the aclnnMatmulGetWorkspaceSize result.
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMatmul(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(mat2_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMatmul failed: ", status));
  }
  aclrtSynchronizeStream(stream);
  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(mat2_tensor);
  aclDestroyTensor(out_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyAclOpExecutor(executor);
  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error MatmulHandlerCublasImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Buffer<ffi::DataType::F32> mat2, ffi::ResultBuffer<ffi::DataType::F32> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);
template ffi::Error MatmulHandlerCublasImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Buffer<ffi::DataType::F16> mat2, ffi::ResultBuffer<ffi::DataType::F16> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);
template ffi::Error MatmulHandlerCublasImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Buffer<ffi::DataType::BF16> mat2, ffi::ResultBuffer<ffi::DataType::BF16> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);

// F32 specialization
ffi::Error MatmulHandlerCublasF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Buffer<ffi::F32> mat2, ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return MatmulHandlerCublasImpl<ffi::DataType::F32>(stream, self, mat2, out, workspace);
}

// F16 specialization
ffi::Error MatmulHandlerCublasF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Buffer<ffi::F16> mat2, ffi::ResultBuffer<ffi::F16> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return MatmulHandlerCublasImpl<ffi::DataType::F16>(stream, self, mat2, out, workspace);
}

// BF16 specialization
ffi::Error MatmulHandlerCublasBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Buffer<ffi::BF16> mat2, ffi::ResultBuffer<ffi::BF16> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return MatmulHandlerCublasImpl<ffi::DataType::BF16>(stream, self, mat2, out, workspace);
}

// Register Matmul operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMatmulCublas,
    MatmulHandlerCublasF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMatmulCublasF32,
    MatmulHandlerCublasF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMatmulCublasF16,
    MatmulHandlerCublasF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMatmulCublasBF16,
    MatmulHandlerCublasBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
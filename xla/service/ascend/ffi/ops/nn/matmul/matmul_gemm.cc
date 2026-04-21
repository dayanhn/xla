#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_gemm.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Gemm operator FFI handler
template <ffi::DataType DType>
ffi::Error GemmHandlerImpl(aclrtStream stream, ffi::Buffer<DType> A, ffi::Buffer<DType> B, ffi::Buffer<DType> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<DType> out, ffi::ResultBuffer<ffi::DataType::S8> workspace) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* A_tensor = ConvertToAclTensor(A);
  aclTensor* B_tensor = ConvertToAclTensor(B);
  aclTensor* C_tensor = ConvertToAclTensor(C);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Gemm operation on stream: " << stream;
  // Set cubeMathType (default to 0 for now)
  int8_t cubeMathType = 0;
  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnGemmGetWorkspaceSize(
      A_tensor, B_tensor, C_tensor, alpha, beta, transA, transB, out_tensor, cubeMathType, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(A_tensor);
    aclDestroyTensor(B_tensor);
    aclDestroyTensor(C_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGemmGetWorkspaceSize failed: ", status));
  }
  // Allocate internal workspace for ascend platform
  // Note: The provided workspace parameter is calculated based on cublas,
  // which is not applicable for ascend platform. We need to allocate
  // our own workspace based on the aclnnGemmGetWorkspaceSize result.
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnGemm(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(A_tensor);
    aclDestroyTensor(B_tensor);
    aclDestroyTensor(C_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGemm failed: ", status));
  }
  aclrtSynchronizeStream(stream);
  // Release resources
  aclDestroyTensor(A_tensor);
  aclDestroyTensor(B_tensor);
  aclDestroyTensor(C_tensor);
  aclDestroyTensor(out_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error GemmHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> A, ffi::Buffer<ffi::DataType::F32> B, ffi::Buffer<ffi::DataType::F32> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::DataType::F32> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);
template ffi::Error GemmHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> A, ffi::Buffer<ffi::DataType::F16> B, ffi::Buffer<ffi::DataType::F16> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::DataType::F16> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);
template ffi::Error GemmHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> A, ffi::Buffer<ffi::DataType::BF16> B, ffi::Buffer<ffi::DataType::BF16> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::DataType::BF16> out, ffi::ResultBuffer<ffi::DataType::S8> workspace);

// F32 specialization
ffi::Error GemmHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> A, ffi::Buffer<ffi::F32> B, ffi::Buffer<ffi::F32> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return GemmHandlerImpl<ffi::DataType::F32>(stream, A, B, C, alpha, beta, transA, transB, out, workspace);
}

// F16 specialization
ffi::Error GemmHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> A, ffi::Buffer<ffi::F16> B, ffi::Buffer<ffi::F16> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::F16> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return GemmHandlerImpl<ffi::DataType::F16>(stream, A, B, C, alpha, beta, transA, transB, out, workspace);
}

// BF16 specialization
ffi::Error GemmHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> A, ffi::Buffer<ffi::BF16> B, ffi::Buffer<ffi::BF16> C, float alpha, float beta, int64_t transA, int64_t transB, ffi::ResultBuffer<ffi::BF16> out, ffi::ResultBuffer<ffi::S8> workspace) {
  return GemmHandlerImpl<ffi::DataType::BF16>(stream, A, B, C, alpha, beta, transA, transB, out, workspace);
}

// Register Gemm operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGemm,
    GemmHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Attr<int64_t>("transA")
        .Attr<int64_t>("transB")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGemmF32,
    GemmHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Attr<int64_t>("transA")
        .Attr<int64_t>("transB")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGemmF16,
    GemmHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Attr<int64_t>("transA")
        .Attr<int64_t>("transB")
        .Ret<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGemmBF16,
    GemmHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Attr<int64_t>("transA")
        .Attr<int64_t>("transB")
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi

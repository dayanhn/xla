#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cast.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <iostream>

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Cast operator FFI handler
template <ffi::DataType SrcDType, ffi::DataType DstDType>
ffi::Error CastHandlerImpl(
    aclrtStream stream, 
    ffi::Buffer<SrcDType> input, 
    ffi::ResultBuffer<DstDType> output) {
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* output_tensor = ConvertToAclTensor(*output);

  // Get aclDataType for destination type using the utility function
  aclDataType dst_acl_type = ConvertToAclDataType(DstDType);

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus status = aclnnCastGetWorkspaceSize(
      input_tensor, dst_acl_type, output_tensor, &workspace_size, &executor);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCastGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_status != ACL_SUCCESS) {
      aclDestroyTensor(input_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  }

  // Call aclnnCast
  status = aclnnCast(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnCast failed: ", status));
  }
  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  return ffi::Error::Success();
}

// Explicit instantiations for supported data type conversions
template ffi::Error CastHandlerImpl<ffi::DataType::S32, ffi::DataType::U32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> input, ffi::ResultBuffer<ffi::DataType::U32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::U8, ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> input, ffi::ResultBuffer<ffi::DataType::S32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::F32, ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> input, ffi::ResultBuffer<ffi::DataType::F16> output);
template ffi::Error CastHandlerImpl<ffi::DataType::F16, ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> input, ffi::ResultBuffer<ffi::DataType::F32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::F32, ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> input, ffi::ResultBuffer<ffi::DataType::BF16> output);
template ffi::Error CastHandlerImpl<ffi::DataType::BF16, ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> input, ffi::ResultBuffer<ffi::DataType::F32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::S32, ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> input, ffi::ResultBuffer<ffi::DataType::F32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::F32, ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> input, ffi::ResultBuffer<ffi::DataType::S32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::U32, ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::U32> input, ffi::ResultBuffer<ffi::DataType::F32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::F32, ffi::DataType::U32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> input, ffi::ResultBuffer<ffi::DataType::U32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::S8, ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S8> input, ffi::ResultBuffer<ffi::DataType::S32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::S32, ffi::DataType::S8>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> input, ffi::ResultBuffer<ffi::DataType::S8> output);
template ffi::Error CastHandlerImpl<ffi::DataType::U8, ffi::DataType::U32>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> input, ffi::ResultBuffer<ffi::DataType::U32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::U32, ffi::DataType::U8>(aclrtStream stream, ffi::Buffer<ffi::DataType::U32> input, ffi::ResultBuffer<ffi::DataType::U8> output);
template ffi::Error CastHandlerImpl<ffi::DataType::PRED, ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::PRED> input, ffi::ResultBuffer<ffi::DataType::S32> output);
template ffi::Error CastHandlerImpl<ffi::DataType::S32, ffi::DataType::PRED>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> input, ffi::ResultBuffer<ffi::DataType::PRED> output);

// S32 to U32 specialization
ffi::Error CastS32ToU32Handler(aclrtStream stream, ffi::Buffer<ffi::S32> input, ffi::ResultBuffer<ffi::U32> output) {
  return CastHandlerImpl<ffi::DataType::S32, ffi::DataType::U32>(stream, input, output);
}

// U8 to S32 specialization
ffi::Error CastU8ToS32Handler(aclrtStream stream, ffi::Buffer<ffi::U8> input, ffi::ResultBuffer<ffi::S32> output) {
  return CastHandlerImpl<ffi::DataType::U8, ffi::DataType::S32>(stream, input, output);
}

// F32 to F16 specialization
ffi::Error CastF32ToF16Handler(aclrtStream stream, ffi::Buffer<ffi::F32> input, ffi::ResultBuffer<ffi::F16> output) {
  return CastHandlerImpl<ffi::DataType::F32, ffi::DataType::F16>(stream, input, output);
}

// F16 to F32 specialization
ffi::Error CastF16ToF32Handler(aclrtStream stream, ffi::Buffer<ffi::F16> input, ffi::ResultBuffer<ffi::F32> output) {
  return CastHandlerImpl<ffi::DataType::F16, ffi::DataType::F32>(stream, input, output);
}

// F32 to BF16 specialization
ffi::Error CastF32ToBF16Handler(aclrtStream stream, ffi::Buffer<ffi::F32> input, ffi::ResultBuffer<ffi::BF16> output) {
  return CastHandlerImpl<ffi::DataType::F32, ffi::DataType::BF16>(stream, input, output);
}

// BF16 to F32 specialization
ffi::Error CastBF16ToF32Handler(aclrtStream stream, ffi::Buffer<ffi::BF16> input, ffi::ResultBuffer<ffi::F32> output) {
  return CastHandlerImpl<ffi::DataType::BF16, ffi::DataType::F32>(stream, input, output);
}

// S32 to F32 specialization
ffi::Error CastS32ToF32Handler(aclrtStream stream, ffi::Buffer<ffi::S32> input, ffi::ResultBuffer<ffi::F32> output) {
  return CastHandlerImpl<ffi::DataType::S32, ffi::DataType::F32>(stream, input, output);
}

// F32 to S32 specialization
ffi::Error CastF32ToS32Handler(aclrtStream stream, ffi::Buffer<ffi::F32> input, ffi::ResultBuffer<ffi::S32> output) {
  return CastHandlerImpl<ffi::DataType::F32, ffi::DataType::S32>(stream, input, output);
}

// U32 to F32 specialization
ffi::Error CastU32ToF32Handler(aclrtStream stream, ffi::Buffer<ffi::U32> input, ffi::ResultBuffer<ffi::F32> output) {
  return CastHandlerImpl<ffi::DataType::U32, ffi::DataType::F32>(stream, input, output);
}

// F32 to U32 specialization
ffi::Error CastF32ToU32Handler(aclrtStream stream, ffi::Buffer<ffi::F32> input, ffi::ResultBuffer<ffi::U32> output) {
  return CastHandlerImpl<ffi::DataType::F32, ffi::DataType::U32>(stream, input, output);
}

// S8 to S32 specialization
ffi::Error CastS8ToS32Handler(aclrtStream stream, ffi::Buffer<ffi::S8> input, ffi::ResultBuffer<ffi::S32> output) {
  return CastHandlerImpl<ffi::DataType::S8, ffi::DataType::S32>(stream, input, output);
}

// S32 to S8 specialization
ffi::Error CastS32ToS8Handler(aclrtStream stream, ffi::Buffer<ffi::S32> input, ffi::ResultBuffer<ffi::S8> output) {
  return CastHandlerImpl<ffi::DataType::S32, ffi::DataType::S8>(stream, input, output);
}

// U8 to U32 specialization
ffi::Error CastU8ToU32Handler(aclrtStream stream, ffi::Buffer<ffi::U8> input, ffi::ResultBuffer<ffi::U32> output) {
  return CastHandlerImpl<ffi::DataType::U8, ffi::DataType::U32>(stream, input, output);
}

// U32 to U8 specialization
ffi::Error CastU32ToU8Handler(aclrtStream stream, ffi::Buffer<ffi::U32> input, ffi::ResultBuffer<ffi::U8> output) {
  return CastHandlerImpl<ffi::DataType::U32, ffi::DataType::U8>(stream, input, output);
}

// BOOL to S32 specialization
ffi::Error CastBoolToS32Handler(aclrtStream stream, ffi::Buffer<ffi::PRED> input, ffi::ResultBuffer<ffi::S32> output) {
  return CastHandlerImpl<ffi::DataType::PRED, ffi::DataType::S32>(stream, input, output);
}

// S32 to BOOL specialization
ffi::Error CastS32ToBoolHandler(aclrtStream stream, ffi::Buffer<ffi::S32> input, ffi::ResultBuffer<ffi::PRED> output) {
  return CastHandlerImpl<ffi::DataType::S32, ffi::DataType::PRED>(stream, input, output);
}

// Register Cast operator FFI functions

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS32ToU32,
    CastS32ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register U8 to S32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastU8ToS32,
    CastU8ToS32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register F32 to F16 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastF32ToF16,
    CastF32ToF16Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register F16 to F32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastF16ToF32,
    CastF16ToF32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register F32 to BF16 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastF32ToBF16,
    CastF32ToBF16Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register BF16 to F32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastBF16ToF32,
    CastBF16ToF32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register S32 to F32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS32ToF32,
    CastS32ToF32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register F32 to S32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastF32ToS32,
    CastF32ToS32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register U32 to F32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastU32ToF32,
    CastU32ToF32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register F32 to U32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastF32ToU32,
    CastF32ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register S8 to S32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS8ToS32,
    CastS8ToS32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register S32 to S8 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS32ToS8,
    CastS32ToS8Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register U8 to U32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastU8ToU32,
    CastU8ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register U32 to U8 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastU32ToU8,
    CastU32ToU8Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Ret<ffi::Buffer<ffi::U8>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register BOOL to S32 Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastBoolToS32,
    CastBoolToS32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Register S32 to BOOL Cast operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS32ToBool,
    CastS32ToBoolHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Generic Cast handler (default to S32 to U32)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCast,
    CastS32ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi

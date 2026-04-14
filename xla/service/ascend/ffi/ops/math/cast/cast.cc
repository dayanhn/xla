#include <iostream>
#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cast.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Cast operator FFI handler for s32 to u32
ffi::Error CastS32ToU32Handler(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> input, 
    ffi::ResultBuffer<ffi::U32> output) {
  
  std::cerr << "[CastS32ToU32] START - Input shape: [";
  for (size_t i = 0; i < input.dimensions().size(); ++i) {
    std::cerr << input.dimensions()[i];
    if (i < input.dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "], Output shape: [";
  for (size_t i = 0; i < output->dimensions().size(); ++i) {
    std::cerr << output->dimensions()[i];
    if (i < output->dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "]" << std::endl;

  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* output_tensor = ConvertToAclTensor(*output);
  
  std::cerr << "[CastS32ToU32] Tensors created - input_tensor: " << input_tensor 
            << ", output_tensor: " << output_tensor << std::endl;

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  std::cerr << "[CastS32ToU32] Calling aclnnCastGetWorkspaceSize..." << std::endl;
  aclnnStatus status = aclnnCastGetWorkspaceSize(
      input_tensor, ACL_UINT32, output_tensor, &workspace_size, &executor);
  
  std::cerr << "[CastS32ToU32] aclnnCastGetWorkspaceSize returned - status: " << status 
            << ", workspace_size: " << workspace_size 
            << ", executor: " << executor << std::endl;
  
  if (status != ACL_SUCCESS) {
    std::cerr << "[CastS32ToU32] ERROR: aclnnCastGetWorkspaceSize failed with status: " << status << std::endl;
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCastGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    std::cerr << "[CastS32ToU32] Allocating workspace memory, size: " << workspace_size << std::endl;
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    std::cerr << "[CastS32ToU32] aclrtMalloc returned - status: " << alloc_status 
              << ", workspaceAddr: " << workspaceAddr << std::endl;
    if (alloc_status != ACL_SUCCESS) {
      std::cerr << "[CastS32ToU32] ERROR: aclrtMalloc failed with status: " << alloc_status << std::endl;
      aclDestroyTensor(input_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  } else {
    std::cerr << "[CastS32ToU32] No workspace memory needed" << std::endl;
  }

  // Call aclnnCast
  std::cerr << "[CastS32ToU32] Calling aclnnCast..." << std::endl;
  status = aclnnCast(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  std::cerr << "[CastS32ToU32] aclnnCast returned - status: " << status << std::endl;
  if (status != ACL_SUCCESS) {
    std::cerr << "[CastS32ToU32] ERROR: aclnnCast failed with status: " << status << std::endl;
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnCast failed: ", status));
  }

  // Synchronize stream
  std::cerr << "[CastS32ToU32] Calling aclrtSynchronizeStream..." << std::endl;
  status = aclrtSynchronizeStream(stream);
  std::cerr << "[CastS32ToU32] aclrtSynchronizeStream returned - status: " << status << std::endl;
  if(status != ACL_SUCCESS){
    std::cerr << "[CastS32ToU32] ERROR: aclrtSynchronizeStream failed with status: " << status << std::endl;
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  std::cerr << "[CastS32ToU32] Releasing resources..." << std::endl;
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  std::cerr << "[CastS32ToU32] FINISHED - Success" << std::endl;
  return ffi::Error::Success();
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

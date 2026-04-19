#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_right_shift.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <vector>

namespace ffi = xla::ffi;

namespace xla::ffi {

// RightShift operator FFI handler
ffi::Error RightShiftHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> input, 
    int32_t shift_bits_value,
    ffi::ResultBuffer<ffi::S32> output) {
  
  // Convert XLA Buffer to Ascend Tensor for input and output
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* output_tensor = ConvertToAclTensor(*output);

  // Create a scalar tensor for shift_bits (1x1 tensor for broadcasting)
  void* shift_bits_device_addr = nullptr;
  aclError alloc_status = aclrtMalloc(&shift_bits_device_addr, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  if (alloc_status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclrtMalloc for shift_bits failed: ", alloc_status));
  }
  
  // Copy the scalar value to device
  alloc_status = aclrtMemcpy(shift_bits_device_addr, sizeof(int32_t), 
                             &shift_bits_value, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (alloc_status != ACL_SUCCESS) {
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclrtMemcpy for shift_bits failed: ", alloc_status));
  }
  
  // Create 1x1 tensor for shift_bits (scalar with shape {})
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> scalar_strides = {};
  aclTensor* shift_bits_tensor = aclCreateTensor(
      scalar_shape.data(), 
      scalar_shape.size(), 
      ACL_INT32, 
      scalar_strides.data(), 
      0, 
      ACL_FORMAT_ND, 
      scalar_shape.data(), 
      scalar_shape.size(), 
      shift_bits_device_addr);
  
  if (shift_bits_tensor == nullptr) {
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal("Failed to create shift_bits aclTensor");
  }

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus status = aclnnRightShiftGetWorkspaceSize(
      input_tensor, shift_bits_tensor, output_tensor, &workspace_size, &executor);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(shift_bits_tensor);
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRightShiftGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_status != ACL_SUCCESS) {
      aclDestroyTensor(shift_bits_tensor);
      aclrtFree(shift_bits_device_addr);
      aclDestroyTensor(input_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  }

  // Call aclnnRightShift
  status = aclnnRightShift(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(shift_bits_tensor);
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnRightShift failed: ", status));
  }

  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyTensor(shift_bits_tensor);
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensor(shift_bits_tensor);
  aclrtFree(shift_bits_device_addr);
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  //aclDestroyAclOpExecutor(executor);

  return ffi::Error::Success();
}

// Register RightShift operator FFI function

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendRightShift,
    RightShiftHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int32_t>("shift_bits")
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
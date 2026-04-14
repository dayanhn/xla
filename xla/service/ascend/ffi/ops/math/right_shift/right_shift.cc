#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_right_shift.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <vector>
#include <iostream>

namespace ffi = xla::ffi;

namespace xla::ffi {

// RightShift operator FFI handler
ffi::Error RightShiftHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> input, 
    int32_t shift_bits_value,
    ffi::ResultBuffer<ffi::S32> output) {
  
  std::cerr << "[RightShift] START - Input shape: [";
  for (size_t i = 0; i < input.dimensions().size(); ++i) {
    std::cerr << input.dimensions()[i];
    if (i < input.dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "], shift_bits_value: " << shift_bits_value << std::endl;
  
  // Convert XLA Buffer to Ascend Tensor for input and output
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* output_tensor = ConvertToAclTensor(*output);
  
  std::cerr << "[RightShift] Tensors created - input_tensor: " << input_tensor 
            << ", output_tensor: " << output_tensor << std::endl;

  // Create a scalar tensor for shift_bits (1x1 tensor for broadcasting)
  void* shift_bits_device_addr = nullptr;
  std::cerr << "[RightShift] Allocating memory for shift_bits scalar..." << std::endl;
  aclError alloc_status = aclrtMalloc(&shift_bits_device_addr, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  std::cerr << "[RightShift] aclrtMalloc returned - status: " << alloc_status 
            << ", addr: " << shift_bits_device_addr << std::endl;
  if (alloc_status != ACL_SUCCESS) {
    std::cerr << "[RightShift] ERROR: Failed to allocate memory for shift_bits scalar, status: " << alloc_status << std::endl;
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclrtMalloc for shift_bits failed: ", alloc_status));
  }
  
  // Copy the scalar value to device
  std::cerr << "[RightShift] Copying shift_bits value (" << shift_bits_value << ") to device..." << std::endl;
  alloc_status = aclrtMemcpy(shift_bits_device_addr, sizeof(int32_t), 
                             &shift_bits_value, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  std::cerr << "[RightShift] aclrtMemcpy returned - status: " << alloc_status << std::endl;
  if (alloc_status != ACL_SUCCESS) {
    std::cerr << "[RightShift] ERROR: Failed to copy shift_bits to device, status: " << alloc_status << std::endl;
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclrtMemcpy for shift_bits failed: ", alloc_status));
  }
  
  // Create 1x1 tensor for shift_bits (scalar with shape {})
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> scalar_strides = {};
  std::cerr << "[RightShift] Creating shift_bits tensor..." << std::endl;
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
  
  std::cerr << "[RightShift] shift_bits_tensor created: " << shift_bits_tensor << std::endl;
  if (shift_bits_tensor == nullptr) {
    std::cerr << "[RightShift] ERROR: Failed to create shift_bits tensor" << std::endl;
    aclrtFree(shift_bits_device_addr);
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal("Failed to create shift_bits aclTensor");
  }

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  std::cerr << "[RightShift] Calling aclnnRightShiftGetWorkspaceSize..." << std::endl;
  aclnnStatus status = aclnnRightShiftGetWorkspaceSize(
      input_tensor, shift_bits_tensor, output_tensor, &workspace_size, &executor);
  
  std::cerr << "[RightShift] aclnnRightShiftGetWorkspaceSize returned - status: " << status 
            << ", workspace_size: " << workspace_size 
            << ", executor: " << executor << std::endl;
  
  if (status != ACL_SUCCESS) {
    std::cerr << "[RightShift] ERROR: aclnnRightShiftGetWorkspaceSize failed with status: " << status << std::endl;
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
    std::cerr << "[RightShift] Allocating workspace memory, size: " << workspace_size << std::endl;
    alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    std::cerr << "[RightShift] aclrtMalloc for workspace returned - status: " << alloc_status 
              << ", addr: " << workspaceAddr << std::endl;
    if (alloc_status != ACL_SUCCESS) {
      std::cerr << "[RightShift] ERROR: aclrtMalloc for workspace failed with status: " << alloc_status << std::endl;
      aclDestroyTensor(shift_bits_tensor);
      aclrtFree(shift_bits_device_addr);
      aclDestroyTensor(input_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  } else {
    std::cerr << "[RightShift] No workspace memory needed" << std::endl;
  }

  // Call aclnnRightShift
  std::cerr << "[RightShift] Calling aclnnRightShift..." << std::endl;
  status = aclnnRightShift(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  std::cerr << "[RightShift] aclnnRightShift returned - status: " << status << std::endl;
  if (status != ACL_SUCCESS) {
    std::cerr << "[RightShift] ERROR: aclnnRightShift failed with status: " << status << std::endl;
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
  std::cerr << "[RightShift] Calling aclrtSynchronizeStream..." << std::endl;
  status = aclrtSynchronizeStream(stream);
  std::cerr << "[RightShift] aclrtSynchronizeStream returned - status: " << status << std::endl;
  if(status != ACL_SUCCESS){
    std::cerr << "[RightShift] ERROR: aclrtSynchronizeStream failed with status: " << status << std::endl;
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
  std::cerr << "[RightShift] Releasing resources..." << std::endl;
  aclDestroyTensor(shift_bits_tensor);
  aclrtFree(shift_bits_device_addr);
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  std::cerr << "[RightShift] FINISHED - Success" << std::endl;
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
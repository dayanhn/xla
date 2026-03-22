#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_gelu.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// GELU operator FFI handler
ffi::Error GeluHandler(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::ResultBuffer<ffi::F32> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for GELU operation on stream: " << stream ;

#if 0  
  // Print device addresses for input and output
  LOG(INFO) << "GeluHandler input buffer device address: " << self.untyped_data();
  LOG(INFO) << "GeluHandler output buffer device address: " << out->untyped_data();
 
  // Print input data
  absl::Status print_status = PrintTensorFirstNElements(self_tensor, 16, "input");
  if (!print_status.ok()) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(std::string(print_status.message()));
  }
#endif

#if 1
  // Temporarily comment out ACLNN call and directly set output to 5.0
  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnGeluGetWorkspaceSize(
      self_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGeluGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnGelu(
      nullptr,  // workspace is managed by XLA
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGelu failed: ", status));
  }
#else
  // Get output tensor address
  void* out_data = nullptr;
  aclError acl_status = aclGetRawTensorAddr(out_tensor, &out_data);
  if (acl_status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        std::string(absl::StrCat("aclGetRawTensorAddr failed for out_tensor: ", acl_status)));
  }

  // Calculate total elements (assuming 32x32 tensor for testing)
  int total_elements = 32 * 32;

  // Allocate host memory and set all values to 5.0
  float* host_data = nullptr;
  acl_status = aclrtMallocHost((void**)&host_data, total_elements * sizeof(float));
  if (acl_status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        std::string(absl::StrCat("aclrtMallocHost failed: ", acl_status)));
  }

  // Set all values to 5.0
  for (int i = 0; i < total_elements; i++) {
    host_data[i] = 5.0f;
  }

  // Copy data from host to device
  acl_status = aclrtMemcpy(out_data, total_elements * sizeof(float),
                          host_data, total_elements * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_status != ACL_SUCCESS) {
    aclrtFreeHost(host_data);
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        std::string(absl::StrCat("aclrtMemcpy failed: ", acl_status)));
  }

  // Free host memory
  aclrtFreeHost(host_data);


  // Print output data
  print_status = PrintTensorFirstNElements(out_tensor, 16, "output");
  if (!print_status.ok()) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(std::string(print_status.message()));
  }
#endif

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);

  return ffi::Error::Success();
}

// Register GELU operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGelu,
    GeluHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
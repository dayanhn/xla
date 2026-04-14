#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_foreach_zero_inplace.h"
#include "third_party/acl/inc/aclnnop/aclnn_foreach_add_scalar_v2.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Full operator FFI handler for F32
ffi::Error FullHandlerF32(
    aclrtStream stream, 
    ffi::Buffer<ffi::F32> self,
    float value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create tensor list with single tensor
  std::vector<aclTensor*> temp_tensors{self_tensor};
  aclTensorList* tensor_list = aclCreateTensorList(temp_tensors.data(), temp_tensors.size());
  if (tensor_list == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }
  
  LOG(INFO) << "Full F32: value=" << value;

  // Step 1: Call aclnnForeachZeroInplace to zero out the tensor
  uint64_t workspace_size_zero = 0;
  aclOpExecutor* executor_zero = nullptr;
  aclnnStatus status = aclnnForeachZeroInplaceGetWorkspaceSize(
      tensor_list, &workspace_size_zero, &executor_zero);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplaceGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr_zero = nullptr;
  if (workspace_size_zero > 0) {
    aclrtMalloc(&workspaceAddr_zero, workspace_size_zero, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachZeroInplace(
      workspaceAddr_zero,
      workspace_size_zero,
      executor_zero,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplace failed: ", status));
  }

  // Step 2: Call aclnnForeachAddScalarV2 to add the scalar value
  // Create scalar value
  aclScalar* scalar_value = aclCreateScalar(&value, ACL_FLOAT);
  if (scalar_value == nullptr) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclScalar");
  }
  
  LOG(INFO) << "Calling aclnnForeachAddScalarV2 to add value " << value << " on stream: " << stream;

  uint64_t workspace_size_add = 0;
  aclOpExecutor* executor_add = nullptr;
  status = aclnnForeachAddScalarV2GetWorkspaceSize(
      tensor_list, scalar_value, tensor_list, &workspace_size_add, &executor_add);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2GetWorkspaceSize failed: ", status));
  }

  void* workspaceAddr_add = nullptr;
  if (workspace_size_add > 0) {
    aclrtMalloc(&workspaceAddr_add, workspace_size_add, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachAddScalarV2(
      workspaceAddr_add,  // workspace is managed by XLA
      workspace_size_add,
      executor_add,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2 failed: ", status));
  }

  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyScalar(scalar_value);
  aclDestroyTensorList(tensor_list);
  aclDestroyTensor(self_tensor);
  if (workspace_size_zero > 0) {
    aclrtFree(workspaceAddr_zero);
  }
  if (workspace_size_add > 0) {
    aclrtFree(workspaceAddr_add);
  }

  return ffi::Error::Success();
}

// Full operator FFI handler for S32
ffi::Error FullHandlerS32(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> self,
    int32_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create tensor list with single tensor
  std::vector<aclTensor*> temp_tensors{self_tensor};
  aclTensorList* tensor_list = aclCreateTensorList(temp_tensors.data(), temp_tensors.size());
  if (tensor_list == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }
  
  LOG(INFO) << "Full S32: value=" << value;

  // Step 1: Call aclnnForeachZeroInplace to zero out the tensor
  uint64_t workspace_size_zero = 0;
  aclOpExecutor* executor_zero = nullptr;
  aclnnStatus status = aclnnForeachZeroInplaceGetWorkspaceSize(
      tensor_list, &workspace_size_zero, &executor_zero);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplaceGetWorkspaceSize failed: ", status));
  }
  // Allocate workspace memory for zero operation
  void* workspaceAddr_zero = nullptr;
  if (workspace_size_zero > 0) {
    aclrtMalloc(&workspaceAddr_zero, workspace_size_zero, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachZeroInplace(
      workspaceAddr_zero,
      workspace_size_zero,
      executor_zero,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplace failed: ", status));
  }

  // Step 2: Call aclnnForeachAddScalarV2 to add the scalar value
  // Create scalar value
  aclScalar* scalar_value = aclCreateScalar(&value, ACL_INT32);
  if (scalar_value == nullptr) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal("Failed to create aclScalar");
  }
  
  LOG(INFO) << "Calling aclnnForeachAddScalarV2 to add value " << value << " on stream: " << stream;

  uint64_t workspace_size_add = 0;
  aclOpExecutor* executor_add = nullptr;
  status = aclnnForeachAddScalarV2GetWorkspaceSize(
      tensor_list, scalar_value, tensor_list, &workspace_size_add, &executor_add);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2GetWorkspaceSize failed: ", status));
  }

  // Allocate workspace memory for add operation
  void* workspaceAddr_add = nullptr;
  if (workspace_size_add > 0) {
    aclrtMalloc(&workspaceAddr_add, workspace_size_add, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachAddScalarV2(
      workspaceAddr_add,
      workspace_size_add,
      executor_add,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    if (workspace_size_add > 0) {
      aclrtFree(workspaceAddr_add);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2 failed: ", status));
  }

  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    if (workspace_size_add > 0) {
      aclrtFree(workspaceAddr_add);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyScalar(scalar_value);
  aclDestroyTensorList(tensor_list);
  aclDestroyTensor(self_tensor);
  if (workspace_size_zero > 0) {
    aclrtFree(workspaceAddr_zero);
  }
  if (workspace_size_add > 0) {
    aclrtFree(workspaceAddr_add);
  }

  return ffi::Error::Success();
}

// Full operator FFI handler for S64
ffi::Error FullHandlerS64(
    aclrtStream stream, 
    ffi::Buffer<ffi::S64> self,
    int64_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create tensor list with single tensor
  std::vector<aclTensor*> temp_tensors{self_tensor};
  aclTensorList* tensor_list = aclCreateTensorList(temp_tensors.data(), temp_tensors.size());
  if (tensor_list == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }
  
  LOG(INFO) << "Full S64: value=" << value;

  // Step 1: Call aclnnForeachZeroInplace to zero out the tensor
  uint64_t workspace_size_zero = 0;
  aclOpExecutor* executor_zero = nullptr;
  aclnnStatus status = aclnnForeachZeroInplaceGetWorkspaceSize(
      tensor_list, &workspace_size_zero, &executor_zero);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplaceGetWorkspaceSize failed: ", status));
  }
  // Allocate workspace memory for zero operation
  void* workspaceAddr_zero = nullptr;
  if (workspace_size_zero > 0) {
    aclrtMalloc(&workspaceAddr_zero, workspace_size_zero, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachZeroInplace(
      workspaceAddr_zero,
      workspace_size_zero,
      executor_zero,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachZeroInplace failed: ", status));
  }

  // Step 2: Call aclnnForeachAddScalarV2 to add the scalar value
  // Create scalar value
  aclScalar* scalar_value = aclCreateScalar(&value, ACL_INT64);
  if (scalar_value == nullptr) {
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal("Failed to create aclScalar");
  }
  
  LOG(INFO) << "Calling aclnnForeachAddScalarV2 to add value " << value << " on stream: " << stream;

  uint64_t workspace_size_add = 0;
  aclOpExecutor* executor_add = nullptr;
  status = aclnnForeachAddScalarV2GetWorkspaceSize(
      tensor_list, scalar_value, tensor_list, &workspace_size_add, &executor_add);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2GetWorkspaceSize failed: ", status));
  }

  // Allocate workspace memory for add operation
  void* workspaceAddr_add = nullptr;
  if (workspace_size_add > 0) {
    aclrtMalloc(&workspaceAddr_add, workspace_size_add, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  status = aclnnForeachAddScalarV2(
      workspaceAddr_add,
      workspace_size_add,
      executor_add,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    if (workspace_size_add > 0) {
      aclrtFree(workspaceAddr_add);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnForeachAddScalarV2 failed: ", status));
  }

  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyScalar(scalar_value);
    aclDestroyTensorList(tensor_list);
    aclDestroyTensor(self_tensor);
    if (workspace_size_zero > 0) {
      aclrtFree(workspaceAddr_zero);
    }
    if (workspace_size_add > 0) {
      aclrtFree(workspaceAddr_add);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyScalar(scalar_value);
  aclDestroyTensorList(tensor_list);
  aclDestroyTensor(self_tensor);
  if (workspace_size_zero > 0) {
    aclrtFree(workspaceAddr_zero);
  }
  if (workspace_size_add > 0) {
    aclrtFree(workspaceAddr_add);
  }

  return ffi::Error::Success();
}

// Register Full operator FFI functions for different data types

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendFullF32,
    FullHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendFullS32,
    FullHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int32_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendFullS64,
    FullHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<int64_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

// Generic handler for Full (default to F32)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendFull,
    FullHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("value"),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi

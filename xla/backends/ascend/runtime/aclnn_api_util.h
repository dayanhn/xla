/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_API_UTIL_H_
#define XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_API_UTIL_H_

#include "absl/log/check.h"
#include "xla/stream_executor/stream.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include <dlfcn.h>
#include <functional>
#include <type_traits>
#include <vector>

// Forward declarations of ACL data structures
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

// Function type definitions for ACL APIs
typedef aclTensor *(*_aclCreateTensor)(
    const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
    const int64_t *stride, int64_t offset, aclFormat format,
    const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value,
                                               uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value,
                                               uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);

// Map XLA PrimitiveType to ACL DataType
constexpr aclDataType PrimitiveTypeToAclDataType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::U8: return ACL_UINT8;
    case xla::PrimitiveType::S8: return ACL_INT8;
    case xla::PrimitiveType::U16: return ACL_UINT16;
    case xla::PrimitiveType::S16: return ACL_INT16;
    case xla::PrimitiveType::U32: return ACL_UINT32;
    case xla::PrimitiveType::S32: return ACL_INT32;
    case xla::PrimitiveType::U64: return ACL_UINT64;
    case xla::PrimitiveType::S64: return ACL_INT64;
    case xla::PrimitiveType::F16: return ACL_FLOAT16;
    case xla::PrimitiveType::BF16: return ACL_BF16;
    case xla::PrimitiveType::F32: return ACL_FLOAT;
    case xla::PrimitiveType::F64: return ACL_DOUBLE;
    case xla::PrimitiveType::PRED: return ACL_BOOL;
    case xla::PrimitiveType::C64: return ACL_COMPLEX64;
    case xla::PrimitiveType::C128: return ACL_COMPLEX128;
    default: return ACL_DT_UNDEFINED;
  }
}

// Macro to get ACL API functions
#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

// Convert XLA BufferAllocation::Slice to aclTensor
inline aclTensor *ConvertType(const xla::gpu::BufferAllocations& buffer_allocations, const xla::BufferAllocation::Slice& slice, const xla::Shape& shape) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  // Get device address from buffer allocations
  auto device_addr = buffer_allocations.GetDeviceAddress(slice);
  if (!device_addr.opaque()) {
    return nullptr;
  }

  // Get data type
  xla::PrimitiveType primitive_type = shape.element_type();
  aclDataType acl_data_type = PrimitiveTypeToAclDataType(primitive_type);
  CHECK(acl_data_type != ACL_DT_UNDEFINED) << "Unsupported data type: " << xla::PrimitiveType_Name(primitive_type);

  // Get dimensions
  std::vector<int64_t> dimensions;
  for (int64_t dim : shape.dimensions()) {
    dimensions.push_back(dim);
  }

  // Calculate strides (assuming row-major)
  std::vector<int64_t> strides(dimensions.size(), 1);
  for (int i = dimensions.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dimensions[i + 1];
  }

  // Create aclTensor
  return aclCreateTensor(
      dimensions.data(),
      dimensions.size(),
      acl_data_type,
      strides.data(),
      0,  // offset
      ACL_FORMAT_ND,
      dimensions.data(),
      dimensions.size(),
      const_cast<void*>(device_addr.opaque()));
}

// Convert scalar value to aclScalar
template <typename T>
inline aclScalar *ConvertType(const T& value, xla::PrimitiveType type) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }

  aclDataType acl_data_type = PrimitiveTypeToAclDataType(type);
  CHECK(acl_data_type != ACL_DT_UNDEFINED) << "Unsupported data type: " << xla::PrimitiveType_Name(type);

  return aclCreateScalar(const_cast<void*>(reinterpret_cast<const void*>(&value)), acl_data_type);
}

// Convert integer array to aclIntArray
inline aclIntArray *ConvertType(const std::vector<int64_t>& values) {
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }

  return aclCreateIntArray(values.data(), values.size());
}

// Convert boolean array to aclBoolArray
inline aclBoolArray *ConvertType(const std::vector<bool>& values) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  // Convert std::vector<bool> to bool array
  std::vector<bool> bool_values(values.begin(), values.end());
  return aclCreateBoolArray(bool_values.data(), bool_values.size());
}

// Convert tensor list to aclTensorList
inline aclTensorList *ConvertType(const std::vector<aclTensor*>& tensors) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  return aclCreateTensorList(tensors.data(), tensors.size());
}

// Template specialization for primitive types
inline aclDataType ConvertType(xla::PrimitiveType type) {
  return PrimitiveTypeToAclDataType(type);
}

// Template fallback for other types
template <typename T>
inline T ConvertType(T value) {
  return value;
}

// Release functions for ACL resources
inline void Release(aclTensor *p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }
  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }
  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p) {
  static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }
  aclDestroyTensorList(p);
}

// Template fallback for other types
template <typename T>
inline void Release(T value) {
  (void)value;
}

// Helper function to release a tuple of converted types
template <typename Tuple, size_t... I>
void ReleaseTuple(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  ReleaseTuple(t, std::make_index_sequence<size>{});
}

// Helper function to release a vector of pointers
template <typename T>
void ReleaseVector(const std::vector<T*>& vec) {
  for (auto* item : vec) {
    Release(item);
  }
}

// Helper function to check ACL status
inline void CheckAclStatus(aclError status, const char* message) {
  if (status != ACL_SUCCESS) {
    CHECK(false) << message << ": " << aclGetRecentErrMsg();
  }
}

// Helper function to get ACL format from XLA shape
inline aclFormat GetAclFormatFromShape(const xla::Shape& shape) {
  int rank = shape.dimensions_size();
  switch (rank) {
    case 0:
      return ACL_FORMAT_ND;
    case 1:
      return ACL_FORMAT_ND;
    case 2:
      return ACL_FORMAT_ND;
    case 3:
      return ACL_FORMAT_NCL;
    case 4:
      return ACL_FORMAT_NCHW;
    case 5:
      return ACL_FORMAT_NCDHW;
    default:
      return ACL_FORMAT_ND;
  }
}

// Helper function to calculate strides for a given shape
inline std::vector<int64_t> CalculateStrides(const std::vector<int64_t>& dimensions) {
  std::vector<int64_t> strides(dimensions.size(), 1);
  for (int i = dimensions.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dimensions[i + 1];
  }
  return strides;
}

inline const char *GetOpApiLibName(void) { return "libopapi.so"; }

inline const char *GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName,
                                   const char *apiName) {
  auto funcAddr = dlsym(handler, apiName);
  return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  return handler;
}

inline void *GetOpApiFuncAddr(const char *apiName) {
  static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
  if (custOpApiHandler != nullptr) {
    auto funcAddr =
        GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

// Function type definitions
typedef void (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);

// Helper function to convert parameters to a tuple
template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args) {
  return std::make_tuple(ConvertType(args)...);
}

// Helper function to call a function with a tuple of arguments
template <typename Function, typename Tuple, size_t... I>
auto CallFunction(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto CallFunction(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return CallFunction(f, t, std::make_index_sequence<size>{});
}

// Helper function to convert a tuple to op api function
template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void *opApiAddr, std::index_sequence<I...>) {
  typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

// Macro to execute aclnn commands
#define EXEC_ACLNN_CMD(aclnn_api, ...)                                          \
  do {                                                                        
    static const auto getWorkspaceSizeFuncAddr =                              
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           
    static const auto initMemAddr =                                           
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           
    static const auto unInitMemAddr =                                         
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    
    CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr)    
        << #aclnn_api << " or " << #aclnn_api << "GetWorkspaceSize not found in aclnn library";
    aclrtStream acl_stream = static_cast<aclrtStream>(stream->platform_specific_handle().stream); \
    uint64_t workspace_size = 0;                                              
    uint64_t *workspace_size_addr = &workspace_size;                          
    aclOpExecutor *executor = nullptr;                                        
    aclOpExecutor **executor_addr = &executor;                                
    
    // Initialize huge memory thread local if available
    InitHugeMemThreadLocal initMemFunc =                                      
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                
    if (initMemFunc) {                                                        
      initMemFunc(nullptr, false);                                            
    }                                                                         
    
    // Convert parameters
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr); 
    
    // Get workspace size
    static auto getWorkspaceSizeFunc =                                        
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       
    auto workspace_status = CallFunction(getWorkspaceSizeFunc, converted_params); 
    CHECK(workspace_status == ACL_SUCCESS)                                   
        << "call " << #aclnn_api << "GetWorkspaceSize failed: " << aclGetRecentErrMsg(); 
    
    // Allocate workspace if needed
    void *workspace_addr = nullptr;                                           
    if (workspace_size > 0) {                                                
      aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST); 
    }                                                                         
    
    // Call the ACLNN API
    typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream); 
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);         
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream); 
    CHECK(api_ret == ACL_SUCCESS)                                            
        << "call " << #aclnn_api << " failed: " << aclGetRecentErrMsg();      
    
    // Release resources
    if (workspace_size > 0) {                                                
      aclrtFree(workspace_addr);                                             
    }                                                                         
    ReleaseConvertTypes(converted_params);                                    
    
    // Release huge memory if available
    ReleaseHugeMem releaseMemFunc =                                           
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                     
    if (releaseMemFunc) {                                                    
      releaseMemFunc(nullptr, false);                                         
    }                                                                         
    
    // Uninitialize huge memory thread local if available
    UnInitHugeMemThreadLocal unInitMemFunc =                                  
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            
    if (unInitMemFunc) {                                                     
      unInitMemFunc(nullptr, false);                                         
    }                                                                         
  } while (false)

#endif  // XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_API_UTIL_H_
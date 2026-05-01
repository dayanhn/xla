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
#include <optional>

namespace xla {
namespace ascend {

// ACLNN_CHECK macro: prints error message and returns InvalidArgumentError on failure
#define ACLNN_CHECK(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "ACLNN_CHECK failed: " << (message) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return absl::InvalidArgumentError(message); \
    } \
  } while (false)



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
constexpr aclDataType PrimitiveTypeToAclDataType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::U8: return ACL_UINT8;
    case PrimitiveType::S8: return ACL_INT8;
    case PrimitiveType::U16: return ACL_UINT16;
    case PrimitiveType::S16: return ACL_INT16;
    case PrimitiveType::U32: return ACL_UINT32;
    case PrimitiveType::S32: return ACL_INT32;
    case PrimitiveType::U64: return ACL_UINT64;
    case PrimitiveType::S64: return ACL_INT64;
    case PrimitiveType::F16: return ACL_FLOAT16;
    case PrimitiveType::BF16: return ACL_BF16;
    case PrimitiveType::F32: return ACL_FLOAT;
    case PrimitiveType::F64: return ACL_DOUBLE;
    case PrimitiveType::PRED: return ACL_BOOL;
    case PrimitiveType::C64: return ACL_COMPLEX64;
    case PrimitiveType::C128: return ACL_COMPLEX128;
    default: return ACL_DT_UNDEFINED;
  }
}

// Forward declarations for OpAPI functions
inline const char *GetOpApiLibName(void);
inline const char *GetCustOpApiLibName(void);
inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName);
inline void *GetOpApiLibHandler(const char *libName);
inline void *GetOpApiFuncAddr(const char *apiName);

// Macro to get ACL API functions
#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

// Convert XLA BufferAllocation::Slice to aclTensor
// Overload with format parameter for specifying ACL tensor format
inline aclTensor *ConvertType(const gpu::BufferAllocations& buffer_allocations, const BufferAllocation::Slice& slice, const Shape& shape, aclFormat format) {
  // Get device address from buffer allocations
  auto device_addr = buffer_allocations.GetDeviceAddress(slice);
  if (!device_addr.opaque()) {
    return nullptr;
  }

  // Get data type
  PrimitiveType primitive_type = shape.element_type();
  aclDataType acl_data_type = PrimitiveTypeToAclDataType(primitive_type);
  CHECK(acl_data_type != ACL_DT_UNDEFINED) << "Unsupported data type: " << PrimitiveType_Name(primitive_type);

  // Get dimensions
  std::vector<int64_t> dimensions;
  for (int64_t dim : shape.dimensions()) {
    dimensions.push_back(dim);
  }

  // Calculate strides based on format
  // For ACL_FORMAT_ND, use standard row-major strides
  // For ACL_FORMAT_NCHW/NCDHW etc., strides depend on the physical memory layout
  std::vector<int64_t> strides(dimensions.size(), 1);
  
  // If format is ACL_FORMAT_ND (or any format that uses row-major), calculate row-major strides
  // Otherwise, the format describes the physical layout, so we use the layout's minor-to-major order
  if (format == ACL_FORMAT_ND) {
    for (int i = dimensions.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dimensions[i + 1];
    }
  } else {
    // For specific formats like NCHW, we need to calculate strides based on the format
    // The format parameter tells ACL how the data is laid out in memory
    // We use default row-major strides for now
    for (int i = dimensions.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dimensions[i + 1];
    }
  }

  // Create aclTensor
  return aclCreateTensor(
      dimensions.data(),
      dimensions.size(),
      acl_data_type,
      strides.data(),
      0,  // offset
      format,
      dimensions.data(),
      dimensions.size(),
      const_cast<void*>(device_addr.opaque()));
}

// Overload without format parameter (defaults to ACL_FORMAT_ND)
inline aclTensor *ConvertType(const gpu::BufferAllocations& buffer_allocations, const BufferAllocation::Slice& slice, const Shape& shape) {
  return ConvertType(buffer_allocations, slice, shape, ACL_FORMAT_ND);
}

// Convert scalar value to aclScalar
template <typename T>
inline aclScalar *ConvertType(const T& value, PrimitiveType type) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }

  aclDataType acl_data_type = PrimitiveTypeToAclDataType(type);
  CHECK(acl_data_type != ACL_DT_UNDEFINED) << "Unsupported data type: " << PrimitiveType_Name(type);

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

  // Convert std::vector<bool> to bool array (std::vector<bool> is specialized and doesn't have .data())
  std::vector<uint8_t> bool_array(values.begin(), values.end());
  return aclCreateBoolArray(reinterpret_cast<const bool*>(bool_array.data()), bool_array.size());
}

// Convert tensor list to aclTensorList
inline aclTensorList *ConvertType(const std::vector<aclTensor*>& tensors) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  return aclCreateTensorList(tensors.data(), tensors.size());
}

// Convert optional tensor to aclTensor
inline aclTensor *ConvertType(const std::optional<aclTensor*>& opt_tensor) {
  if (opt_tensor.has_value() && *opt_tensor != nullptr) {
    return *opt_tensor;
  }
  return nullptr;
}

// Convert optional int64_t array to aclIntArray
inline aclIntArray *ConvertType(const std::optional<std::vector<int64_t>>& opt_array) {
  if (opt_array.has_value()) {
    return ConvertType(*opt_array);
  }
  return nullptr;
}

// Template specialization for primitive types
inline aclDataType ConvertType(PrimitiveType type) {
  return PrimitiveTypeToAclDataType(type);
}

// Helper struct to wrap tensor triplet (buffer_allocations, slice, shape)
// This allows passing tensor information without immediate conversion
struct TensorTriplet {
  const gpu::BufferAllocations* buffer_allocations;
  BufferAllocation::Slice slice;
  Shape shape;
  aclFormat format;  // ACL format hint (defaults to ACL_FORMAT_ND)
  
  TensorTriplet() : buffer_allocations(nullptr), format(ACL_FORMAT_ND) {}
  
  TensorTriplet(const gpu::BufferAllocations* ba, const BufferAllocation::Slice& s, const Shape& sh, aclFormat fmt = ACL_FORMAT_ND)
      : buffer_allocations(ba), slice(s), shape(sh), format(fmt) {}
};

// Helper function to convert TensorTriplet to aclTensor*
inline aclTensor* ConvertType(const TensorTriplet& triplet) {
  return ConvertType(*triplet.buffer_allocations, triplet.slice, triplet.shape, triplet.format);
}


// Handle aclTensor* (including nullptr case)
inline aclTensor* ConvertType(aclTensor* tensor) {
  return tensor;  // Directly pass through, whether it's nullptr or valid pointer
}

// Handle nullptr for optional tensor parameters - convert to aclTensor*
inline aclTensor* ConvertType(std::nullptr_t) {
  return nullptr;
}

// Pass-through ConvertType for pointer types that don't need conversion
inline uint64_t* ConvertType(uint64_t* value) {
  return value;
}

// Pass-through ConvertType for aclOpExecutor**
inline aclOpExecutor** ConvertType(aclOpExecutor** value) {
  return value;
}

// Pass-through ConvertType for float
inline float ConvertType(float value) {
  return value;
}

// Pass-through ConvertType for int64_t
inline int64_t ConvertType(int64_t value) {
  return value;
}

// Pass-through ConvertType for bool
inline bool ConvertType(bool value) {
  return value;
}

// Pass-through ConvertType for int8_t
inline int8_t ConvertType(int8_t value) {
  return value;
}

// Release functions for ACL resources
inline void Release(aclTensor *p) {
  if(p == nullptr) return;
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  if(p == nullptr) return;
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  if(p == nullptr) return;
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }
  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  if(p == nullptr) return;
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }
  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p) {
  if(p == nullptr) return;
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
  int rank = shape.dimensions().size();
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
auto ConvertTypes(Ts&&... args) -> decltype(std::make_tuple(ConvertType(std::forward<Ts>(args))...)) {
  return std::make_tuple(ConvertType(std::forward<Ts>(args))...);
}

// Helper function to call a function with a tuple of arguments
template <typename Function, typename Tuple, size_t... I>
auto CallFunction(Function f, Tuple& t, std::index_sequence<I...>) -> decltype(f(std::get<I>(t)...)) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto CallFunction(Function f, Tuple& t) -> decltype(CallFunction(f, t, std::make_index_sequence<std::tuple_size<Tuple>::value>{})) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return CallFunction(f, t, std::make_index_sequence<size>{});
}

// Helper function to convert a tuple to op api function
template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void *opApiAddr, std::index_sequence<I...>) {
#if 0  
  // Debug: Print parameter types for troubleshooting
  std::cerr << "[ConvertToOpApiFunc] Converting function at addr: " << opApiAddr << std::endl;
  std::cerr << "[ConvertToOpApiFunc] Number of parameters: " << sizeof...(I) << std::endl;
  
  // Print each parameter type info (using typeid for runtime type info)
  int param_idx = 0;
  auto print_type = [&](auto idx) {
    constexpr size_t Index = decltype(idx)::value;
    using ParamType = typename std::decay<decltype(std::get<Index>(params))>::type;
    std::cerr << "[ConvertToOpApiFunc]   Param[" << param_idx++ << "] type: " 
              << typeid(ParamType).name() << std::endl;
  };
  
  // Expand parameter pack to print all types
  (print_type(std::integral_constant<size_t, I>{}), ...);
#endif  
  typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  
  //std::cerr << "[ConvertToOpApiFunc] Converted function pointer: " 
  //          << reinterpret_cast<void*>(func) << std::endl;
  
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

// Helper macro to convert (buffer_allocations, slice, shape) triplet to aclTensor*
#define CONVERT_TENSOR_TRIPLET(buffer_allocs, slice, shape) \
  ConvertType(buffer_allocs, slice, shape)

// Macro to execute aclnn commands
#define EXEC_ACLNN_CMD(aclnn_api, stream, ...)                                \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    ACLNN_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, \
        std::string(#aclnn_api) + " or " + #aclnn_api +                      \
        "GetWorkspaceSize not found in aclnn library");                      \
    aclrtStream acl_stream = static_cast<aclrtStream>(                        \
        stream->platform_specific_handle().stream);                           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
                                                                              \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
                                                                              \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
                                                                              \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status =                                                   \
        CallFunction(getWorkspaceSizeFunc, converted_params);                 \
    ACLNN_CHECK(workspace_status == ACL_SUCCESS,                              \
        std::string("call ") + #aclnn_api + "GetWorkspaceSize failed: " +    \
        (aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "unknown error"));    \
                                                                              \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size > 0) {                                                 \
      aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);\
    }                                                                         \
                                                                              \
    typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream); \
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);         \
    auto api_ret =                                                            \
        opApiFunc(workspace_addr, workspace_size, executor, acl_stream);      \
    ACLNN_CHECK(api_ret == ACL_SUCCESS,                                       \
        std::string("call ") + #aclnn_api + " failed: " +                    \
        (aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "unknown error"));    \
                                                                              \
    if (workspace_size > 0) {                                                 \
      aclrtFree(workspace_addr);                                              \
    }                                                                         \
    ReleaseConvertTypes(converted_params);                                    \
                                                                              \
    ReleaseHugeMem releaseMemFunc =                                           \
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                     \
    if (releaseMemFunc) {                                                     \
      releaseMemFunc(nullptr, false);                                         \
    }                                                                         \
                                                                              \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
 } while (false)

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_API_UTIL_H_

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
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include <dlfcn.h>
#include <functional>
#include <type_traits>
#include <vector>

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

// Convert parameters to the format expected by aclnn functions
template <typename... Args>
inline std::vector<void*> ConvertTypes(Args&&... args) {
  std::vector<void*> params;
  (params.push_back(const_cast<void*>(reinterpret_cast<const void*>(&args)), ...);
  return params;
}

// Release converted parameters
inline void ReleaseConvertTypes(std::vector<void*>& params) {
  // No-op for now, but might be needed for complex types
}

// Convert to op api function template
template <typename FuncType>
inline FuncType ConvertToOpApiFunc(const std::vector<void*>& params, void* func_addr) {
  return reinterpret_cast<FuncType>(func_addr);
}

// Macro to execute aclnn commands
#define EXEC_ACLNN_CMD(aclnn_api, ...)                                          \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr)    \
        << #aclnn_api << " or " << #aclnn_api << "GetWorkspaceSize not found in aclnn library";
    aclrtStream acl_stream = static_cast<aclrtStream>(stream->platform_specific_handle().stream); \
    uint64_t workspace_size = 0;                                              \
    aclOpExecutor *executor = nullptr;                                        \
    auto converted_params = ConvertTypes(__VA_ARGS__, &workspace_size, &executor); \
    using GetWorkspaceSizeFuncType = int (*)(void*...);                      \
    static auto getWorkspaceSizeFunc =                                       \
        ConvertToOpApiFunc<GetWorkspaceSizeFuncType>(converted_params, getWorkspaceSizeFuncAddr); \
    auto workspace_status = getWorkspaceSizeFunc(converted_params.data());    \
    CHECK(workspace_status == ACL_SUCCESS)                                   \
        << "call " << #aclnn_api << "GetWorkspaceSize failed: " << aclGetRecentErrMsg(); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size > 0) {                                                \
      aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST); \
    }                                                                         \
    typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream); \
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);         \
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
    CHECK(api_ret == ACL_SUCCESS)                                            \
        << "call " << #aclnn_api << " failed: " << aclGetRecentErrMsg();      \
    if (workspace_size > 0) {                                                \
      aclrtFree(workspace_addr);                                             \
    }                                                                         \
    ReleaseConvertTypes(converted_params);                                    \
  } while (false)

#endif  // XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_API_UTIL_H_

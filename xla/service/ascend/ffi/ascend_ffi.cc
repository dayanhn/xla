#include "xla/service/ascend/ffi/ascend_ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "absl/status/status.h"
#include "absl/log/log.h"

namespace xla::ffi {

void RegisterAscendFfiHandlers() {
  // Register GELU operator
  auto error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gelu",
      "ASCEND",
      AscendGelu);
  
  auto status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gelu operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gelu operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.gelu operator";
  }

  // Register Matmul operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmul",
      "ASCEND",
      AscendMatmul);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.matmul operator";
  }

  // Register InplaceIndexFillTensor operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor",
      "ASCEND",
      AscendInplaceIndexFillTensor);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.inplace_index_fill_tensor operator";
  }

  // Register Full operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.full",
      "ASCEND",
      AscendFull);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.full operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.full operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.full operator";
  }

  // Register FullF32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.full.f32",
      "ASCEND",
      AscendFullF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.full.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.full.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.full.f32 operator";
  }

  // Register FullS32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.full.s32",
      "ASCEND",
      AscendFullS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.full.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.full.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.full.s32 operator";
  }

  // Register FullS64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.full.s64",
      "ASCEND",
      AscendFullS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.full.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.full.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.full.s64 operator";
  }

  // Register Cast S32 to U32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.s32_to_u32",
      "ASCEND",
      AscendCastS32ToU32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.s32_to_u32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.s32_to_u32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.cast.s32_to_u32 operator";
  }

  // Register Cast operator (default to S32 to U32)
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast",
      "ASCEND",
      AscendCast);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.cast operator";
  }

  // Register RightShift operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.right_shift",
      "ASCEND",
      AscendRightShift);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.right_shift operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.right_shift operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.right_shift operator";
  }

  // Register Cat operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cat",
      "ASCEND",
      AscendCat);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cat operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cat operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.cat operator";
  }

  // Register other operators here in the future
}

}  // namespace xla::ffi

namespace {
// 模块初始化时注册
static bool InitModule() {
  try {
    xla::ffi::RegisterAscendFfiHandlers();
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to initialize Ascend FFI module: " << e.what();
    return false;
  }
}

static bool module_initialized = InitModule();

}  // namespace
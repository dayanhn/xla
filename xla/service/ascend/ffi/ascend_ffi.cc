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

  // Register Add operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add",
      "ASCEND",
      AscendAdd);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add operator";
  }

  // Register Add F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add.f32",
      "ASCEND",
      AscendAddF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add.f32 operator";
  }

  // Register Add F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add.f16",
      "ASCEND",
      AscendAddF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add.f16 operator";
  }

  // Register Add BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add.bf16",
      "ASCEND",
      AscendAddBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add.bf16 operator";
  }

  // Register Add S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add.s32",
      "ASCEND",
      AscendAddS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add.s32 operator";
  }

  // Register Add S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.add.s64",
      "ASCEND",
      AscendAddS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.add.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.add.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.add.s64 operator";
  }

  // Register Divide operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide",
      "ASCEND",
      AscendDivide);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide operator";
  }

  // Register Divide F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide.f32",
      "ASCEND",
      AscendDivideF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide.f32 operator";
  }

  // Register Divide F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide.f16",
      "ASCEND",
      AscendDivideF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide.f16 operator";
  }

  // Register Divide BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide.bf16",
      "ASCEND",
      AscendDivideBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide.bf16 operator";
  }

  // Register Divide S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide.s32",
      "ASCEND",
      AscendDivideS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide.s32 operator";
  }

  // Register Divide S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.divide.s64",
      "ASCEND",
      AscendDivideS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.divide.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.divide.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.divide.s64 operator";
  }

  // Register Equal operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal",
      "ASCEND",
      AscendEqual);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal operator";
  }

  // Register Equal F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.f32",
      "ASCEND",
      AscendEqualF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.f32 operator";
  }

  // Register Equal F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.f16",
      "ASCEND",
      AscendEqualF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.f16 operator";
  }

  // Register Equal BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.bf16",
      "ASCEND",
      AscendEqualBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.bf16 operator";
  }

  // Register Equal S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.s32",
      "ASCEND",
      AscendEqualS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.s32 operator";
  }

  // Register Equal S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.s64",
      "ASCEND",
      AscendEqualS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.s64 operator";
  }

  // Register Equal U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.u8",
      "ASCEND",
      AscendEqualU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.u8 operator";
  }

  // Register Equal S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.s8",
      "ASCEND",
      AscendEqualS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.s8 operator";
  }

  // Register Equal BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.equal.bool",
      "ASCEND",
      AscendEqualPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.equal.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.equal.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.equal.bool operator";
  }

  // Register Exponential operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential",
      "ASCEND",
      AscendExponential);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential operator";
  }

  // Register Exponential F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential.f32",
      "ASCEND",
      AscendExponentialF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential.f32 operator";
  }

  // Register Exponential F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential.f16",
      "ASCEND",
      AscendExponentialF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential.f16 operator";
  }

  // Register Exponential BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential.bf16",
      "ASCEND",
      AscendExponentialBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential.bf16 operator";
  }

  // Register Exponential S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential.s32",
      "ASCEND",
      AscendExponentialS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential.s32 operator";
  }

  // Register Exponential S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.exponential.s64",
      "ASCEND",
      AscendExponentialS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.exponential.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.exponential.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.exponential.s64 operator";
  }

  // Register Expand operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand",
      "ASCEND",
      AscendExpand);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand operator";
  }

  // Register Expand F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.f32",
      "ASCEND",
      AscendExpandF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.f32 operator";
  }

  // Register Expand F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.f16",
      "ASCEND",
      AscendExpandF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.f16 operator";
  }

  // Register Expand BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.bf16",
      "ASCEND",
      AscendExpandBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.bf16 operator";
  }

  // Register Expand S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.s32",
      "ASCEND",
      AscendExpandS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.s32 operator";
  }

  // Register Expand S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.s64",
      "ASCEND",
      AscendExpandS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.s64 operator";
  }

  // Register Expand U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.u8",
      "ASCEND",
      AscendExpandU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.u8 operator";
  }

  // Register Expand S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.s8",
      "ASCEND",
      AscendExpandS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.s8 operator";
  }

  // Register Expand BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.expand.bool",
      "ASCEND",
      AscendExpandPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.expand.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.expand.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.expand.bool operator";
  }

  // Register Greater operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater",
      "ASCEND",
      AscendGreater);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater operator";
  }

  // Register Greater F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.f32",
      "ASCEND",
      AscendGreaterF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.f32 operator";
  }

  // Register Greater F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.f16",
      "ASCEND",
      AscendGreaterF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.f16 operator";
  }

  // Register Greater BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.bf16",
      "ASCEND",
      AscendGreaterBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.bf16 operator";
  }

  // Register Greater S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.s32",
      "ASCEND",
      AscendGreaterS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.s32 operator";
  }

  // Register Greater S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.s64",
      "ASCEND",
      AscendGreaterS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.s64 operator";
  }

  // Register Greater U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.u8",
      "ASCEND",
      AscendGreaterU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.u8 operator";
  }

  // Register Greater S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.s8",
      "ASCEND",
      AscendGreaterS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.s8 operator";
  }

  // Register Greater BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater.bool",
      "ASCEND",
      AscendGreaterPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater.bool operator";
  }

  // Register GreaterEqual operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal",
      "ASCEND",
      AscendGreaterEqual);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal operator";
  }

  // Register GreaterEqual F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.f32",
      "ASCEND",
      AscendGreaterEqualF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.f32 operator";
  }

  // Register GreaterEqual F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.f16",
      "ASCEND",
      AscendGreaterEqualF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.f16 operator";
  }

  // Register GreaterEqual BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.bf16",
      "ASCEND",
      AscendGreaterEqualBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.bf16 operator";
  }

  // Register GreaterEqual S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.s32",
      "ASCEND",
      AscendGreaterEqualS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.s32 operator";
  }

  // Register GreaterEqual S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.s64",
      "ASCEND",
      AscendGreaterEqualS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.s64 operator";
  }

  // Register GreaterEqual U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.u8",
      "ASCEND",
      AscendGreaterEqualU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.u8 operator";
  }

  // Register GreaterEqual S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.s8",
      "ASCEND",
      AscendGreaterEqualS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.s8 operator";
  }

  // Register GreaterEqual BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.greater_equal.bool",
      "ASCEND",
      AscendGreaterEqualPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.greater_equal.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.greater_equal.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.greater_equal.bool operator";
  }

  // Register Less operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less",
      "ASCEND",
      AscendLess);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less operator";
  }

  // Register Less F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.f32",
      "ASCEND",
      AscendLessF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.f32 operator";
  }

  // Register Less F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.f16",
      "ASCEND",
      AscendLessF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.f16 operator";
  }

  // Register Less BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.bf16",
      "ASCEND",
      AscendLessBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.bf16 operator";
  }

  // Register Less S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.s32",
      "ASCEND",
      AscendLessS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.s32 operator";
  }

  // Register Less S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.s64",
      "ASCEND",
      AscendLessS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.s64 operator";
  }

  // Register Less U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.u8",
      "ASCEND",
      AscendLessU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.u8 operator";
  }

  // Register Less S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.s8",
      "ASCEND",
      AscendLessS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.s8 operator";
  }

  // Register Less BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less.bool",
      "ASCEND",
      AscendLessPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less.bool operator";
  }

  // Register LessEqual operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal",
      "ASCEND",
      AscendLessEqual);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal operator";
  }

  // Register LessEqual F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.f32",
      "ASCEND",
      AscendLessEqualF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.f32 operator";
  }

  // Register LessEqual F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.f16",
      "ASCEND",
      AscendLessEqualF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.f16 operator";
  }

  // Register LessEqual BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.bf16",
      "ASCEND",
      AscendLessEqualBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.bf16 operator";
  }

  // Register LessEqual S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.s32",
      "ASCEND",
      AscendLessEqualS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.s32 operator";
  }

  // Register LessEqual S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.s64",
      "ASCEND",
      AscendLessEqualS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.s64 operator";
  }

  // Register LessEqual U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.u8",
      "ASCEND",
      AscendLessEqualU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.u8 operator";
  }

  // Register LessEqual S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.s8",
      "ASCEND",
      AscendLessEqualS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.s8 operator";
  }

  // Register LessEqual BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.less_equal.bool",
      "ASCEND",
      AscendLessEqualPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.less_equal.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.less_equal.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.less_equal.bool operator";
  }

  // Register Maximum operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum",
      "ASCEND",
      AscendMaximum);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum operator";
  }

  // Register Maximum F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum.f32",
      "ASCEND",
      AscendMaximumF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum.f32 operator";
  }

  // Register Maximum F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum.f16",
      "ASCEND",
      AscendMaximumF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum.f16 operator";
  }

  // Register Maximum BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum.bf16",
      "ASCEND",
      AscendMaximumBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum.bf16 operator";
  }

  // Register Maximum S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum.s32",
      "ASCEND",
      AscendMaximumS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum.s32 operator";
  }

  // Register Maximum S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.maximum.s64",
      "ASCEND",
      AscendMaximumS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.maximum.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.maximum.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.maximum.s64 operator";
  }

  // Register Multiply operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply",
      "ASCEND",
      AscendMultiply);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply operator";
  }

  // Register Multiply F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply.f32",
      "ASCEND",
      AscendMultiplyF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply.f32 operator";
  }

  // Register Multiply F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply.f16",
      "ASCEND",
      AscendMultiplyF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply.f16 operator";
  }

  // Register Multiply BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply.bf16",
      "ASCEND",
      AscendMultiplyBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply.bf16 operator";
  }

  // Register Multiply S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply.s32",
      "ASCEND",
      AscendMultiplyS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply.s32 operator";
  }

  // Register Multiply S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.multiply.s64",
      "ASCEND",
      AscendMultiplyS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.multiply.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.multiply.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.multiply.s64 operator";
  }

  // Register Negate operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate",
      "ASCEND",
      AscendNegate);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate operator";
  }

  // Register Negate F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate.f32",
      "ASCEND",
      AscendNegateF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate.f32 operator";
  }

  // Register Negate F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate.f16",
      "ASCEND",
      AscendNegateF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate.f16 operator";
  }

  // Register Negate BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate.bf16",
      "ASCEND",
      AscendNegateBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate.bf16 operator";
  }

  // Register Negate S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate.s32",
      "ASCEND",
      AscendNegateS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate.s32 operator";
  }

  // Register Negate S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.negate.s64",
      "ASCEND",
      AscendNegateS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.negate.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.negate.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.negate.s64 operator";
  }

  // Register NotEqual operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal",
      "ASCEND",
      AscendNotEqual);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal operator";
  }

  // Register NotEqual F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.f32",
      "ASCEND",
      AscendNotEqualF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.f32 operator";
  }

  // Register NotEqual F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.f16",
      "ASCEND",
      AscendNotEqualF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.f16 operator";
  }

  // Register NotEqual BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.bf16",
      "ASCEND",
      AscendNotEqualBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.bf16 operator";
  }

  // Register NotEqual S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.s32",
      "ASCEND",
      AscendNotEqualS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.s32 operator";
  }

  // Register NotEqual S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.s64",
      "ASCEND",
      AscendNotEqualS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.s64 operator";
  }

  // Register NotEqual U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.u8",
      "ASCEND",
      AscendNotEqualU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.u8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.u8 operator";
  }

  // Register NotEqual S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.s8",
      "ASCEND",
      AscendNotEqualS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.s8 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.s8 operator";
  }

  // Register NotEqual BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.not_equal.bool",
      "ASCEND",
      AscendNotEqualPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.not_equal.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.not_equal.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.not_equal.bool operator";
  }

  // Register ReduceMax operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max",
      "ASCEND",
      AscendReduceMax);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max operator";
  }

  // Register ReduceMax F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max.f32",
      "ASCEND",
      AscendReduceMaxF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max.f32 operator";
  }

  // Register ReduceMax F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max.f16",
      "ASCEND",
      AscendReduceMaxF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max.f16 operator";
  }

  // Register ReduceMax BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max.bf16",
      "ASCEND",
      AscendReduceMaxBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max.bf16 operator";
  }

  // Register ReduceMax S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max.s32",
      "ASCEND",
      AscendReduceMaxS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max.s32 operator";
  }

  // Register ReduceMax S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_max.s64",
      "ASCEND",
      AscendReduceMaxS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_max.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_max.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_max.s64 operator";
  }

  // Register ReduceMean operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_mean",
      "ASCEND",
      AscendReduceMean);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_mean operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_mean operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_mean operator";
  }

  // Register ReduceMean F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_mean.f32",
      "ASCEND",
      AscendReduceMeanF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_mean.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_mean.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_mean.f32 operator";
  }

  // Register ReduceMean F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_mean.f16",
      "ASCEND",
      AscendReduceMeanF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_mean.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_mean.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_mean.f16 operator";
  }

  // Register ReduceMean BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_mean.bf16",
      "ASCEND",
      AscendReduceMeanBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_mean.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_mean.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_mean.bf16 operator";
  }

  // Register ReduceMin operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min",
      "ASCEND",
      AscendReduceMin);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min operator";
  }

  // Register ReduceMin F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min.f32",
      "ASCEND",
      AscendReduceMinF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min.f32 operator";
  }

  // Register ReduceMin F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min.f16",
      "ASCEND",
      AscendReduceMinF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min.f16 operator";
  }

  // Register ReduceMin BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min.bf16",
      "ASCEND",
      AscendReduceMinBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min.bf16 operator";
  }

  // Register ReduceMin S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min.s32",
      "ASCEND",
      AscendReduceMinS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min.s32 operator";
  }

  // Register ReduceMin S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_min.s64",
      "ASCEND",
      AscendReduceMinS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_min.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_min.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_min.s64 operator";
  }

  // Register ReduceProd operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod",
      "ASCEND",
      AscendReduceProd);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod operator";
  }

  // Register ReduceProd F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod.f32",
      "ASCEND",
      AscendReduceProdF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod.f32 operator";
  }

  // Register ReduceProd F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod.f16",
      "ASCEND",
      AscendReduceProdF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod.f16 operator";
  }

  // Register ReduceProd BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod.bf16",
      "ASCEND",
      AscendReduceProdBf16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod.bf16 operator";
  }

  // Register ReduceProd S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod.s32",
      "ASCEND",
      AscendReduceProdS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod.s32 operator";
  }

  // Register ReduceProd S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_prod.s64",
      "ASCEND",
      AscendReduceProdS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_prod.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_prod.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_prod.s64 operator";
  }

  // Register Select operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select",
      "ASCEND",
      AscendSelect);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select operator";
  }

  // Register Select F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.f32",
      "ASCEND",
      AscendSelectF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.f32 operator";
  }

  // Register Select F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.f16",
      "ASCEND",
      AscendSelectF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.f16 operator";
  }

  // Register Select BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.bf16",
      "ASCEND",
      AscendSelectBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.bf16 operator";
  }

  // Register Select S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.s32",
      "ASCEND",
      AscendSelectS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.s32 operator";
  }

  // Register Select S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.s64",
      "ASCEND",
      AscendSelectS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.s64 operator";
  }

  // Register Select BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select.bool",
      "ASCEND",
      AscendSelectPRED);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select.bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select.bool operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select.bool operator";
  }

  // Register Subtract operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract",
      "ASCEND",
      AscendSubtract);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract operator";
  }

  // Register Subtract F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract.f32",
      "ASCEND",
      AscendSubtractF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract.f32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract.f32 operator";
  }

  // Register Subtract F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract.f16",
      "ASCEND",
      AscendSubtractF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract.f16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract.f16 operator";
  }

  // Register Subtract BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract.bf16",
      "ASCEND",
      AscendSubtractBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract.bf16 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract.bf16 operator";
  }

  // Register Subtract S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract.s32",
      "ASCEND",
      AscendSubtractS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract.s32 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract.s32 operator";
  }

  // Register Subtract S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.subtract.s64",
      "ASCEND",
      AscendSubtractS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.subtract.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.subtract.s64 operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.subtract.s64 operator";
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

  // Register ReduceSum operators
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum",
      "ASCEND",
      AscendReduceSum);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.reduce_sum operator";
  }

  // Register Select operators
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.select",
      "ASCEND",
      AscendSelect);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.select operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.select operator: " + status.ToString());
  }else {
    LOG(INFO) << "Registered ascend.select operator";
  }
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
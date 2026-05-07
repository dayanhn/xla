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
  }

  // Register Cast U8 to S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.u8_to_s32",
      "ASCEND",
      AscendCastU8ToS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.u8_to_s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.u8_to_s32 operator: " + status.ToString());
  }

  // Register Cast F32 to F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.f32_to_f16",
      "ASCEND",
      AscendCastF32ToF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.f32_to_f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.f32_to_f16 operator: " + status.ToString());
  }

  // Register Cast F16 to F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.f16_to_f32",
      "ASCEND",
      AscendCastF16ToF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.f16_to_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.f16_to_f32 operator: " + status.ToString());
  }

  // Register Cast F32 to BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.f32_to_bf16",
      "ASCEND",
      AscendCastF32ToBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.f32_to_bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.f32_to_bf16 operator: " + status.ToString());
  }

  // Register Cast BF16 to F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.bf16_to_f32",
      "ASCEND",
      AscendCastBF16ToF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.bf16_to_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.bf16_to_f32 operator: " + status.ToString());
  }

  // Register Cast S32 to F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.s32_to_f32",
      "ASCEND",
      AscendCastS32ToF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.s32_to_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.s32_to_f32 operator: " + status.ToString());
  }

  // Register Cast F32 to S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.f32_to_s32",
      "ASCEND",
      AscendCastF32ToS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.f32_to_s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.f32_to_s32 operator: " + status.ToString());
  }

  // Register Cast U32 to F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.u32_to_f32",
      "ASCEND",
      AscendCastU32ToF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.u32_to_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.u32_to_f32 operator: " + status.ToString());
  }

  // Register Cast F32 to U32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.f32_to_u32",
      "ASCEND",
      AscendCastF32ToU32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.f32_to_u32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.f32_to_u32 operator: " + status.ToString());
  }

  // Register Cast S8 to S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.s8_to_s32",
      "ASCEND",
      AscendCastS8ToS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.s8_to_s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.s8_to_s32 operator: " + status.ToString());
  }

  // Register Cast S32 to S8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.s32_to_s8",
      "ASCEND",
      AscendCastS32ToS8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.s32_to_s8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.s32_to_s8 operator: " + status.ToString());
  }

  // Register Cast U8 to U32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.u8_to_u32",
      "ASCEND",
      AscendCastU8ToU32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.u8_to_u32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.u8_to_u32 operator: " + status.ToString());
  }

  // Register Cast U32 to U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.u32_to_u8",
      "ASCEND",
      AscendCastU32ToU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.u32_to_u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.u32_to_u8 operator: " + status.ToString());
  }

  // Register Cast BOOL to S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.bool_to_s32",
      "ASCEND",
      AscendCastBoolToS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.bool_to_s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.bool_to_s32 operator: " + status.ToString());
  }

  // Register Cast S32 to BOOL operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.cast.s32_to_bool",
      "ASCEND",
      AscendCastS32ToBool);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.cast.s32_to_bool operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.cast.s32_to_bool operator: " + status.ToString());
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
  }

  // Register Muls operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.muls",
      "ASCEND",
      AscendMuls);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.muls operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.muls operator: " + status.ToString());
  }

  // Register Muls F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.muls.f32",
      "ASCEND",
      AscendMulsF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.muls.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.muls.f32 operator: " + status.ToString());
  }

  // Register Muls F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.muls.f16",
      "ASCEND",
      AscendMulsF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.muls.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.muls.f16 operator: " + status.ToString());
  }

  // Register Muls BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.muls.bf16",
      "ASCEND",
      AscendMulsBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.muls.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.muls.bf16 operator: " + status.ToString());
  }

  // Register Muls S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.muls.s32",
      "ASCEND",
      AscendMulsS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.muls.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.muls.s32 operator: " + status.ToString());
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
  }
  // Register Matmul F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmul_f32",
      "ASCEND",
      AscendMatmulF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_f32 operator: " + status.ToString());
  }
  // Register Matmul F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmul_f16",
      "ASCEND",
      AscendMatmulF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_f16 operator: " + status.ToString());
  }
  // Register Matmul BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmul_bf16",
      "ASCEND",
      AscendMatmulBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_bf16 operator: " + status.ToString());
  }

   // Register Matmul operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmulcublas",
      "ASCEND",
      AscendMatmulCublas);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul operator: " + status.ToString());
  }
  // Register Matmul F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmulcublas.f32",
      "ASCEND",
      AscendMatmulCublasF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_f32 operator: " + status.ToString());
  }
  // Register Matmul F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmulcublas.f16",
      "ASCEND",
      AscendMatmulCublasF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_f16 operator: " + status.ToString());
  }
  // Register Matmul BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.matmulcublas.bf16",
      "ASCEND",
      AscendMatmulCublasBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.matmul_bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.matmul_bf16 operator: " + status.ToString());
  }

  // Register Gemm operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gemm",
      "ASCEND",
      AscendGemm);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gemm operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gemm operator: " + status.ToString());
  }
  // Register Gemm F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gemm.f32",
      "ASCEND",
      AscendGemmF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gemm.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gemm.f32 operator: " + status.ToString());
  }
  // Register Gemm F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gemm.f16",
      "ASCEND",
      AscendGemmF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gemm.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gemm.f16 operator: " + status.ToString());
  }
  // Register Gemm BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gemm.bf16",
      "ASCEND",
      AscendGemmBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gemm.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gemm.bf16 operator: " + status.ToString());
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
  }
  // Register InplaceIndexFillTensor F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor.f32",
      "ASCEND",
      AscendInplaceIndexFillTensorF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor.f32 operator: " + status.ToString());
  }
  // Register InplaceIndexFillTensor F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor.f16",
      "ASCEND",
      AscendInplaceIndexFillTensorF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor.f16 operator: " + status.ToString());
  }
  // Register InplaceIndexFillTensor BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor.bf16",
      "ASCEND",
      AscendInplaceIndexFillTensorBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor.bf16 operator: " + status.ToString());
  }
  // Register InplaceIndexFillTensor S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor.s32",
      "ASCEND",
      AscendInplaceIndexFillTensorS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor.s32 operator: " + status.ToString());
  }
  // Register InplaceIndexFillTensor S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.inplace_index_fill_tensor.s64",
      "ASCEND",
      AscendInplaceIndexFillTensorS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.inplace_index_fill_tensor.s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.inplace_index_fill_tensor.s64 operator: " + status.ToString());
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
  }
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
  }
  // Register ReduceSum F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum_f32",
      "ASCEND",
      AscendReduceSumF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum_f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum_f32 operator: " + status.ToString());
  }
  // Register ReduceSum F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum_f16",
      "ASCEND",
      AscendReduceSumF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum_f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum_f16 operator: " + status.ToString());
  }
  // Register ReduceSum BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum_bf16",
      "ASCEND",
      AscendReduceSumBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum_bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum_bf16 operator: " + status.ToString());
  }
  // Register ReduceSum S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum_s32",
      "ASCEND",
      AscendReduceSumS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum_s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum_s32 operator: " + status.ToString());
  }
  // Register ReduceSum S64 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.reduce_sum_s64",
      "ASCEND",
      AscendReduceSumS64);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.reduce_sum_s64 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.reduce_sum_s64 operator: " + status.ToString());
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
  }
  // Register other operators here in the future

  // Register Iota U8 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.iota.u8",
      "ASCEND",
      AscendIotaU8);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.iota.u8 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.iota.u8 operator: " + status.ToString());
  }

  // Register Iota S32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.iota.s32",
      "ASCEND",
      AscendIotaS32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.iota.s32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.iota.s32 operator: " + status.ToString());
  }

  // Register MaxDim operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.max_dim",
      "ASCEND",
      AscendMaxDim);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.max_dim operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.max_dim operator: " + status.ToString());
  }
  // Register MaxDim F32 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.max_dim.f32",
      "ASCEND",
      AscendMaxDimF32);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.max_dim.f32 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.max_dim.f32 operator: " + status.ToString());
  }
  // Register MaxDim F16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.max_dim.f16",
      "ASCEND",
      AscendMaxDimF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.max_dim.f16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.max_dim.f16 operator: " + status.ToString());
  }
  // Register MaxDim BF16 operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.max_dim.bf16",
      "ASCEND",
      AscendMaxDimBF16);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.max_dim.bf16 operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.max_dim.bf16 operator: " + status.ToString());
  }

  // Register Unified ACLNN operator
  error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend_op",
      "ASCEND",
      AscendUnifiedOp);
  
  status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend_op operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend_op operator: " + status.ToString());
  }}

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
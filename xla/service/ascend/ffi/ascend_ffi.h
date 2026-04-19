#ifndef XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_
#define XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_

#include "xla/ffi/api/api.h"

namespace xla::ffi {

// Register all Ascend FFI handlers
void RegisterAscendFfiHandlers();

// Declare external symbols for FFI handlers
extern "C" XLA_FFI_Error* AscendGelu(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMatmul(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMatmulF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMatmulF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMatmulBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensor(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensorF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensorF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensorBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensorS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendInplaceIndexFillTensorS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendFull(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendFullF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendFullS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendFullS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendCast(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendCastS32ToU32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAdd(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAddF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAddF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAddBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAddS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendAddS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivide(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivideF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivideF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivideBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivideS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendDivideS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqual(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendEqualPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponential(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponentialF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponentialF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponentialBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponentialS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExponentialS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpand(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendExpandPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreater(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqual(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendGreaterEqualPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLess(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqual(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendLessEqualPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximum(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximumF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximumF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximumBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximumS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMaximumS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiply(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiplyF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiplyF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiplyBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiplyS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendMultiplyS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegate(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegateF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegateF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegateBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegateS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNegateS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqual(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualU8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualS8(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendNotEqualPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMax(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMaxF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMaxF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMaxBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMaxS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMaxS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMean(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMeanF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMeanF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMeanBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMin(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMinF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMinF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMinBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMinS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceMinS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProd(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProdF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProdF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProdBf16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProdS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceProdS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelect(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSelectPRED(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtract(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtractF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtractF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtractBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtractS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendSubtractS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendRightShift(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendCat(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSum(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSumF32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSumF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSumBF16(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSumS32(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendReduceSumS64(XLA_FFI_CallFrame* frame);
extern "C" XLA_FFI_Error* AscendIotaU8(XLA_FFI_CallFrame* frame);




}  // namespace xla::ffi

#endif  // XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_
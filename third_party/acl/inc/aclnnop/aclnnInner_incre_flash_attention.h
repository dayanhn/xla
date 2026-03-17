
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_INCRE_FLASH_ATTENTION_H_
#define ACLNN_INNER_INCRE_FLASH_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerIncreFlashAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : dynamic
 * value : dynamic
 * pseShiftOptional : optional
 * attenMaskOptional : optional
 * actualSeqLengthsOptional : optional
 * dequantScale1Optional : optional
 * quantScale1Optional : optional
 * dequantScale2Optional : optional
 * quantScale2Optional : optional
 * quantOffset2Optional : optional
 * antiquantScaleOptional : optional
 * antiquantOffsetOptional : optional
 * blockTableOptional : optional
 * kvPaddingSizeOptional : optional
 * numHeads : required
 * scaleValue : optional
 * inputLayoutOptional : optional
 * numKeyValueHeads : optional
 * blockSize : optional
 * innerPrecise : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerIncreFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclTensor *dequantScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *dequantScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *kvPaddingSizeOptional,
    int64_t numHeads,
    double scaleValue,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t blockSize,
    int64_t innerPrecise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnInnerIncreFlashAttentionTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *dequantScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *dequantScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *kvPaddingSizeOptional,
    int64_t numHeads,
    double scaleValue,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t blockSize,
    int64_t innerPrecise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerIncreFlashAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerIncreFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

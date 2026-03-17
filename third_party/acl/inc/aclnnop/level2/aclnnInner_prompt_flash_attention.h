
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_PROMPT_FLASH_ATTENTION_H_
#define ACLNN_INNER_PROMPT_FLASH_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerPromptFlashAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * pseShiftOptional : optional
 * attenMaskOptional : optional
 * actualSeqLengthsOptional : optional
 * actualSeqLengthsKvOptional : optional
 * deqScale1Optional : optional
 * quantScale1Optional : optional
 * deqScale2Optional : optional
 * quantScale2Optional : optional
 * quantOffset2Optional : optional
 * numHeads : required
 * scaleValue : optional
 * preTokens : optional
 * nextTokens : optional
 * inputLayoutOptional : optional
 * numKeyValueHeads : optional
 * sparseMode : optional
 * innerPrecise : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnInnerPromptFlashAttentionTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerPromptFlashAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerPromptFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

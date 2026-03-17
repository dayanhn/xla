
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_NSA_COMPRESS_ATTENTION_H_
#define ACLNN_INNER_NSA_COMPRESS_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerNsaCompressAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * attenMaskOptional : optional
 * actualSeqQlenOptional : optional
 * actualCmpSeqKvlenOptional : optional
 * actualSelSeqKvlenOptional : optional
 * topkMaskOptional : optional
 * scaleValue : optional
 * headNum : required
 * inputLayout : required
 * sparseMode : optional
 * compressBlockSize : required
 * compressStride : required
 * selectBlockSize : required
 * selectBlockCount : required
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * attentionOutOut : required
 * topkIndicesOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNsaCompressAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqQlenOptional,
    const aclIntArray *actualCmpSeqKvlenOptional,
    const aclIntArray *actualSelSeqKvlenOptional,
    const aclTensor *topkMaskOptional,
    double scaleValue,
    int64_t headNum,
    char *inputLayout,
    int64_t sparseMode,
    int64_t compressBlockSize,
    int64_t compressStride,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *attentionOutOut,
    const aclTensor *topkIndicesOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnInnerNsaCompressAttentionTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualCmpSeqKvlenOptional,
    const aclTensor *actualSelSeqKvlenOptional,
    const aclTensor *topkMaskOptional,
    double scaleValue,
    int64_t headNum,
    char *inputLayout,
    int64_t sparseMode,
    int64_t compressBlockSize,
    int64_t compressStride,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *attentionOutOut,
    const aclTensor *topkIndicesOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerNsaCompressAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNsaCompressAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIN_TRANSFORMER_LN_QKV_H_
#define ACLNN_SWIN_TRANSFORMER_LN_QKV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwinTransformerLnQKVGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * beta : required
 * weight : required
 * bias : required
 * headNum : required
 * headDim : required
 * seqLength : required
 * shiftsOptional : optional
 * epsilon : optional
 * queryOutputOut : required
 * keyOutputOut : required
 * valueOutputOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinTransformerLnQKVGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *weight,
    const aclTensor *bias,
    int64_t headNum,
    int64_t headDim,
    int64_t seqLength,
    const aclIntArray *shiftsOptional,
    double epsilon,
    const aclTensor *queryOutputOut,
    const aclTensor *keyOutputOut,
    const aclTensor *valueOutputOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwinTransformerLnQKV
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinTransformerLnQKV(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

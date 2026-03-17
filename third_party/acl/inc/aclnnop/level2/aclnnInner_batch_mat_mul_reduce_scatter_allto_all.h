
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_H_
#define ACLNN_INNER_BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * biasOptional : optional
 * groupEp : required
 * groupTp : required
 * epWorldSize : required
 * tpWorldSize : required
 * yShardType : optional
 * transposeWeight : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    char *groupEp,
    char *groupTp,
    int64_t epWorldSize,
    int64_t tpWorldSize,
    int64_t yShardType,
    bool transposeWeight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerBatchMatMulReduceScatterAlltoAll
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerBatchMatMulReduceScatterAlltoAll(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

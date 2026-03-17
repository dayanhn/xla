
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MATMUL_REDUCE_SCATTER_V2_H_
#define ACLNN_INNER_MATMUL_REDUCE_SCATTER_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMatmulReduceScatterV2GetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * x1ScaleOptional : optional
 * x2ScaleOptional : optional
 * quantScaleOptional : optional
 * group : required
 * reduceOp : required
 * isTransA : optional
 * isTransB : optional
 * commTurn : optional
 * rankSize : optional
 * blockSize : optional
 * groupSize : optional
 * isAmaxOut : optional
 * yDtype : optional
 * commMode : required
 * yOut : required
 * amaxOutOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const aclTensor *quantScaleOptional,
    char *group,
    char *reduceOp,
    bool isTransA,
    bool isTransB,
    int64_t commTurn,
    int64_t rankSize,
    int64_t blockSize,
    int64_t groupSize,
    bool isAmaxOut,
    int64_t yDtype,
    char *commMode,
    const aclTensor *yOut,
    const aclTensor *amaxOutOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMatmulReduceScatterV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulReduceScatterV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

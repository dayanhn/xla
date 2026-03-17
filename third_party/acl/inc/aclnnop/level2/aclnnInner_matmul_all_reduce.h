
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MATMUL_ALL_REDUCE_H_
#define ACLNN_INNER_MATMUL_ALL_REDUCE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMatmulAllReduceGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * x3Optional : optional
 * antiquantScaleOptional : optional
 * antiquantOffsetOptional : optional
 * dequantScaleOptional : optional
 * pertokenScaleOptional : optional
 * commQuantScale1Optional : optional
 * commQuantScale2Optional : optional
 * group : required
 * reduceOpOptional : optional
 * isTransA : optional
 * isTransB : optional
 * commTurn : optional
 * antiquantGroupSize : optional
 * groupSize : optional
 * yDtype : optional
 * commQuantMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllReduceGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *dequantScaleOptional,
    const aclTensor *pertokenScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    char *group,
    char *reduceOpOptional,
    bool isTransA,
    bool isTransB,
    int64_t commTurn,
    int64_t antiquantGroupSize,
    int64_t groupSize,
    int64_t yDtype,
    int64_t commQuantMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMatmulAllReduce
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllReduce(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MATMUL_ALL_REDUCE_ADD_RMS_NORM_H_
#define ACLNN_INNER_MATMUL_ALL_REDUCE_ADD_RMS_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * residual : required
 * gamma : required
 * antiquantScaleOptional : optional
 * antiquantOffsetOptional : optional
 * dequantScaleOptional : optional
 * group : required
 * reduceOpOptional : optional
 * isTransA : optional
 * isTransB : optional
 * commTurn : optional
 * antiquantGroupSize : optional
 * epsilon : optional
 * yOut : required
 * normOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *residual,
    const aclTensor *gamma,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *dequantScaleOptional,
    char *group,
    char *reduceOpOptional,
    bool isTransA,
    bool isTransB,
    int64_t commTurn,
    int64_t antiquantGroupSize,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *normOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMatmulAllReduceAddRmsNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllReduceAddRmsNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

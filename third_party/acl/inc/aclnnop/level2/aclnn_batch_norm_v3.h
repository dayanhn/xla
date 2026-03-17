
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BATCH_NORM_V3_H_
#define ACLNN_BATCH_NORM_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBatchNormV3GetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * runningMeanRef : required
 * runningVarRef : required
 * epsilon : optional
 * momentum : optional
 * isTraining : optional
 * yOut : required
 * runningMeanRef : required
 * runningVarRef : required
 * saveMeanOut : required
 * saveRstdOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchNormV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    aclTensor *runningMeanRef,
    aclTensor *runningVarRef,
    double epsilon,
    double momentum,
    bool isTraining,
    const aclTensor *yOut,
    const aclTensor *saveMeanOut,
    const aclTensor *saveRstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBatchNormV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchNormV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

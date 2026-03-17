
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_DISTRIBUTE_BARRIER_H_
#define ACLNN_INNER_DISTRIBUTE_BARRIER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerDistributeBarrierGetWorkspaceSize
 * parameters :
 * xRefRef : required
 * timeOutOptional : optional
 * elasticInfoOptional : optional
 * group : required
 * worldSize : required
 * xRefRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(
    aclTensor *xRefRef,
    const aclTensor *timeOutOptional,
    const aclTensor *elasticInfoOptional,
    char *group,
    int64_t worldSize,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerDistributeBarrier
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerDistributeBarrier(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

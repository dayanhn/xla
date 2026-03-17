
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_UPDATE_EXPERT_H_
#define ACLNN_INNER_MOE_UPDATE_EXPERT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeUpdateExpertGetWorkspaceSize
 * parameters :
 * expertIds : required
 * eplbTable : required
 * expertScalesOptional : optional
 * pruningThresholdOptional : optional
 * activeMaskOptional : optional
 * localRankId : optional
 * worldSize : optional
 * balanceMode : optional
 * balancedExpertIdsOut : required
 * balancedActiveMaskOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeUpdateExpertGetWorkspaceSize(
    const aclTensor *expertIds,
    const aclTensor *eplbTable,
    const aclTensor *expertScalesOptional,
    const aclTensor *pruningThresholdOptional,
    const aclTensor *activeMaskOptional,
    int64_t localRankId,
    int64_t worldSize,
    int64_t balanceMode,
    const aclTensor *balancedExpertIdsOut,
    const aclTensor *balancedActiveMaskOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeUpdateExpert
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeUpdateExpert(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

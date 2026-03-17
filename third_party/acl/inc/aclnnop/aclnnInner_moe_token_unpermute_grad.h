
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_TOKEN_UNPERMUTE_GRAD_H_
#define ACLNN_INNER_MOE_TOKEN_UNPERMUTE_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize
 * parameters :
 * permutedTokens : required
 * unpermutedTokensGrad : required
 * sortedIndices : required
 * probsOptional : optional
 * paddedMode : optional
 * restoreShapeOptional : optional
 * permutedTokensGradOut : required
 * probsGradOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor *permutedTokens,
    const aclTensor *unpermutedTokensGrad,
    const aclTensor *sortedIndices,
    const aclTensor *probsOptional,
    bool paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor *permutedTokensGradOut,
    const aclTensor *probsGradOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeTokenUnpermuteGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenUnpermuteGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

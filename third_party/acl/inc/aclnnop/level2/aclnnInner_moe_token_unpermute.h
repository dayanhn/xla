
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_TOKEN_UNPERMUTE_H_
#define ACLNN_INNER_MOE_TOKEN_UNPERMUTE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeTokenUnpermuteGetWorkspaceSize
 * parameters :
 * permutedTokens : required
 * sortedIndices : required
 * probsOptional : optional
 * paddedMode : optional
 * restoreShapeOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor *permutedTokens,
    const aclTensor *sortedIndices,
    const aclTensor *probsOptional,
    bool paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeTokenUnpermute
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenUnpermute(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

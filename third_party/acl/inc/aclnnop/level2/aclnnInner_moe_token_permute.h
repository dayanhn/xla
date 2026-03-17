
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_TOKEN_PERMUTE_H_
#define ACLNN_INNER_MOE_TOKEN_PERMUTE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeTokenPermuteGetWorkspaceSize
 * parameters :
 * tokens : required
 * indices : required
 * numOutTokens : optional
 * paddedMode : optional
 * permuteTokensOut : required
 * sortedIndicesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenPermuteGetWorkspaceSize(
    const aclTensor *tokens,
    const aclTensor *indices,
    int64_t numOutTokens,
    bool paddedMode,
    const aclTensor *permuteTokensOut,
    const aclTensor *sortedIndicesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeTokenPermute
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeTokenPermute(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_DISTRIBUTE_BUFFER_RESET_H_
#define ACLNN_INNER_MOE_DISTRIBUTE_BUFFER_RESET_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeDistributeBufferResetGetWorkspaceSize
 * parameters :
 * elasticInfo : required
 * groupEp : required
 * epWorldSize : required
 * needSync : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeBufferResetGetWorkspaceSize(
    const aclTensor *elasticInfo,
    char *groupEp,
    int64_t epWorldSize,
    int64_t needSync,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeDistributeBufferReset
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeBufferReset(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_STACK_GROUP_POINTS_H_
#define ACLNN_STACK_GROUP_POINTS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnStackGroupPointsGetWorkspaceSize
 * parameters :
 * features : required
 * featuresBatchCnt : required
 * indices : required
 * indicesBatchCnt : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStackGroupPointsGetWorkspaceSize(
    const aclTensor *features,
    const aclTensor *featuresBatchCnt,
    const aclTensor *indices,
    const aclTensor *indicesBatchCnt,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnStackGroupPoints
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStackGroupPoints(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

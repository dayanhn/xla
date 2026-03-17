
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GE_GLU_GRAD_V2_H_
#define ACLNN_GE_GLU_GRAD_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGeGluGradV2GetWorkspaceSize
 * parameters :
 * dy : required
 * x : required
 * gelu : required
 * dim : optional
 * approximate : optional
 * activateLeft : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeGluGradV2GetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x,
    const aclTensor *gelu,
    int64_t dim,
    int64_t approximate,
    bool activateLeft,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGeGluGradV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeGluGradV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

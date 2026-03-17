
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GE_GLU_V2_H_
#define ACLNN_GE_GLU_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGeGluV2GetWorkspaceSize
 * parameters :
 * x : required
 * dim : optional
 * approximate : optional
 * activateLeft : optional
 * yOut : required
 * geluOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeGluV2GetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    int64_t approximate,
    bool activateLeft,
    const aclTensor *yOut,
    const aclTensor *geluOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGeGluV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeGluV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

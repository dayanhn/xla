
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_QUANT_UPDATE_SCATTER_V2_H_
#define ACLNN_DYNAMIC_QUANT_UPDATE_SCATTER_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicQuantUpdateScatterV2GetWorkspaceSize
 * parameters :
 * x : required
 * indices : required
 * varRef : required
 * varScaleRef : required
 * varOffsetRef : required
 * varRef : required
 * varScaleRef : required
 * varOffsetRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicQuantUpdateScatterV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *indices,
    aclTensor *varRef,
    aclTensor *varScaleRef,
    aclTensor *varOffsetRef,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicQuantUpdateScatterV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicQuantUpdateScatterV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_QUANT_UPDATE_SCATTER_H_
#define ACLNN_DYNAMIC_QUANT_UPDATE_SCATTER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicQuantUpdateScatterGetWorkspaceSize
 * parameters :
 * varRef : required
 * varScaleRef : required
 * indices : required
 * updates : required
 * smoothScalesOptional : optional
 * reduce : required
 * axis : optional
 * varRef : required
 * varScaleRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicQuantUpdateScatterGetWorkspaceSize(
    aclTensor *varRef,
    aclTensor *varScaleRef,
    const aclTensor *indices,
    const aclTensor *updates,
    const aclTensor *smoothScalesOptional,
    char *reduce,
    int64_t axis,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicQuantUpdateScatter
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicQuantUpdateScatter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

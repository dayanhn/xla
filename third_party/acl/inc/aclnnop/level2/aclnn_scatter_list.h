
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_LIST_H_
#define ACLNN_SCATTER_LIST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterListGetWorkspaceSize
 * parameters :
 * varRef : dynamic
 * indice : required
 * updates : required
 * maskOptional : optional
 * reduceOptional : optional
 * axis : optional
 * varRef : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterListGetWorkspaceSize(
    aclTensorList *varRef,
    const aclTensor *indice,
    const aclTensor *updates,
    const aclTensor *maskOptional,
    char *reduceOptional,
    int64_t axis,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterList
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterList(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

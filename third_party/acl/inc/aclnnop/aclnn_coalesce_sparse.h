
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_COALESCE_SPARSE_H_
#define ACLNN_COALESCE_SPARSE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCoalesceSparseGetWorkspaceSize
 * parameters :
 * uniqueLen : required
 * uniqueIndices : required
 * indices : required
 * values : required
 * newIndicesOut : required
 * newValuesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCoalesceSparseGetWorkspaceSize(
    const aclTensor *uniqueLen,
    const aclTensor *uniqueIndices,
    const aclTensor *indices,
    const aclTensor *values,
    const aclTensor *newIndicesOut,
    const aclTensor *newValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCoalesceSparse
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCoalesceSparse(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

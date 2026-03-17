
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_GROUPED_MAT_MUL_ALL_REDUCE_H_
#define ACLNN_INNER_GROUPED_MAT_MUL_ALL_REDUCE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize
 * parameters :
 * x : dynamic
 * weight : dynamic
 * bias : dynamic
 * groupListOptional : optional
 * splitItem : optional
 * group : required
 * reduceOpOptional : optional
 * commTurn : optional
 * out : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *bias,
    const aclIntArray *groupListOptional,
    int64_t splitItem,
    char *group,
    char *reduceOpOptional,
    int64_t commTurn,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnInnerGroupedMatMulAllReduceTensorGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *bias,
    const aclTensor *groupListOptional,
    int64_t splitItem,
    char *group,
    char *reduceOpOptional,
    int64_t commTurn,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerGroupedMatMulAllReduce
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerGroupedMatMulAllReduce(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

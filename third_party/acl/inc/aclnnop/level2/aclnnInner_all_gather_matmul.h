
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ALL_GATHER_MATMUL_H_
#define ACLNN_INNER_ALL_GATHER_MATMUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerAllGatherMatmulGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * group : required
 * isTransA : optional
 * isTransB : optional
 * gatherIndex : optional
 * commTurn : optional
 * rankSize : optional
 * isGatherOut : optional
 * yOut : required
 * gatherOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAllGatherMatmulGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    char *group,
    bool isTransA,
    bool isTransB,
    int64_t gatherIndex,
    int64_t commTurn,
    int64_t rankSize,
    bool isGatherOut,
    const aclTensor *yOut,
    const aclTensor *gatherOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerAllGatherMatmul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAllGatherMatmul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

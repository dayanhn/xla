
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MATMUL_REDUCE_SCATTER_H_
#define ACLNN_INNER_MATMUL_REDUCE_SCATTER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMatmulReduceScatterGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * group : required
 * reduceOpOptional : optional
 * isTransA : optional
 * isTransB : optional
 * commTurn : optional
 * rankSize : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulReduceScatterGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    char *group,
    char *reduceOpOptional,
    bool isTransA,
    bool isTransB,
    int64_t commTurn,
    int64_t rankSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMatmulReduceScatter
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulReduceScatter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_APPLY_FTRL_H_
#define ACLNN_INNER_APPLY_FTRL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerApplyFtrlGetWorkspaceSize
 * parameters :
 * varRef : required
 * accum : required
 * linear : required
 * grad : required
 * lr : required
 * l1 : required
 * l2 : required
 * lrPower : required
 * varRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerApplyFtrlGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *accum,
    const aclTensor *linear,
    const aclTensor *grad,
    const aclTensor *lr,
    const aclTensor *l1,
    const aclTensor *l2,
    const aclTensor *lrPower,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerApplyFtrl
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerApplyFtrl(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

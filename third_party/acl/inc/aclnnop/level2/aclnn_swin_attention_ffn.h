
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIN_ATTENTION_FFN_H_
#define ACLNN_SWIN_ATTENTION_FFN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwinAttentionFFNGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * bias : required
 * x3Optional : optional
 * shiftsOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinAttentionFFNGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *x3Optional,
    const aclIntArray *shiftsOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwinAttentionFFN
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinAttentionFFN(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

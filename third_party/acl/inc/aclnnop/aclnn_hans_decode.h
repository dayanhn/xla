
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HANS_DECODE_H_
#define ACLNN_HANS_DECODE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHansDecodeGetWorkspaceSize
 * parameters :
 * mantissa : required
 * fixed : required
 * var : required
 * pdf : required
 * reshuff : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHansDecodeGetWorkspaceSize(
    const aclTensor *mantissa,
    const aclTensor *fixed,
    const aclTensor *var,
    const aclTensor *pdf,
    bool reshuff,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHansDecode
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHansDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

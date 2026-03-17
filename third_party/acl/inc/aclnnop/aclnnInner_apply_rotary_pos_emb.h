
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_APPLY_ROTARY_POS_EMB_H_
#define ACLNN_INNER_APPLY_ROTARY_POS_EMB_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerApplyRotaryPosEmbGetWorkspaceSize
 * parameters :
 * queryRef : required
 * keyRef : required
 * cos : required
 * sin : required
 * layout : optional
 * rotaryModeOptional : optional
 * queryRef : required
 * keyRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerApplyRotaryPosEmbGetWorkspaceSize(
    aclTensor *queryRef,
    aclTensor *keyRef,
    const aclTensor *cos,
    const aclTensor *sin,
    int64_t layout,
    char *rotaryModeOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerApplyRotaryPosEmb
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerApplyRotaryPosEmb(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

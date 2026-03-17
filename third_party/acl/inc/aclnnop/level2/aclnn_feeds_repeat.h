
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FEEDS_REPEAT_H_
#define ACLNN_FEEDS_REPEAT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFeedsRepeatGetWorkspaceSize
 * parameters :
 * feeds : required
 * feedsRepeatTimes : required
 * outputFeedsSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFeedsRepeatGetWorkspaceSize(
    const aclTensor *feeds,
    const aclTensor *feedsRepeatTimes,
    int64_t outputFeedsSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFeedsRepeat
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFeedsRepeat(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

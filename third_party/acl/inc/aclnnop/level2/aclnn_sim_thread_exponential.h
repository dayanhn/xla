
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SIM_THREAD_EXPONENTIAL_H_
#define ACLNN_SIM_THREAD_EXPONENTIAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSimThreadExponentialGetWorkspaceSize
 * parameters :
 * selfRef : required
 * count : required
 * lambd : required
 * seed : required
 * offset : required
 * selfRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSimThreadExponentialGetWorkspaceSize(
    aclTensor *selfRef,
    int64_t count,
    double lambd,
    int64_t seed,
    int64_t offset,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSimThreadExponential
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSimThreadExponential(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

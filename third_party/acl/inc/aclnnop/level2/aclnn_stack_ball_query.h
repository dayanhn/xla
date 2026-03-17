
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_STACK_BALL_QUERY_H_
#define ACLNN_STACK_BALL_QUERY_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnStackBallQueryGetWorkspaceSize
 * parameters :
 * xyz : required
 * centerXyz : required
 * xyzBatchCnt : required
 * centerXyzBatchCnt : required
 * maxRadius : required
 * sampleNum : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStackBallQueryGetWorkspaceSize(
    const aclTensor *xyz,
    const aclTensor *centerXyz,
    const aclTensor *xyzBatchCnt,
    const aclTensor *centerXyzBatchCnt,
    double maxRadius,
    int64_t sampleNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnStackBallQuery
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStackBallQuery(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

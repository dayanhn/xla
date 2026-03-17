
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ROTARY_POSITION_EMBEDDING_H_
#define ACLNN_INNER_ROTARY_POSITION_EMBEDDING_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize
 * parameters :
 * x : required
 * cos : required
 * sin : required
 * mode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *cos,
    const aclTensor *sin,
    int64_t mode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerRotaryPositionEmbedding
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerRotaryPositionEmbedding(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

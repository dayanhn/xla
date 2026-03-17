
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_SCALED_MASKED_SOFTMAX_V2_H_
#define ACLNN_INNER_SCALED_MASKED_SOFTMAX_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize
 * parameters :
 * x : required
 * maskOptional : optional
 * scale : optional
 * fixedTriuMask : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *maskOptional,
    double scale,
    bool fixedTriuMask,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerScaledMaskedSoftmaxV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerScaledMaskedSoftmaxV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

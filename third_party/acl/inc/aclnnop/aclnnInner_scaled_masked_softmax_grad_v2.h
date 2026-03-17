
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_SCALED_MASKED_SOFTMAX_GRAD_V2_H_
#define ACLNN_INNER_SCALED_MASKED_SOFTMAX_GRAD_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerScaledMaskedSoftmaxGradV2GetWorkspaceSize
 * parameters :
 * yGrad : required
 * y : required
 * maskOptional : optional
 * scale : optional
 * fixedTriuMask : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerScaledMaskedSoftmaxGradV2GetWorkspaceSize(
    const aclTensor *yGrad,
    const aclTensor *y,
    const aclTensor *maskOptional,
    double scale,
    bool fixedTriuMask,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerScaledMaskedSoftmaxGradV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerScaledMaskedSoftmaxGradV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

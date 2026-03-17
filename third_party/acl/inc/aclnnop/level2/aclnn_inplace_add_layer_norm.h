
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INPLACE_ADD_LAYER_NORM_H_
#define ACLNN_INPLACE_ADD_LAYER_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInplaceAddLayerNormGetWorkspaceSize
 * parameters :
 * x1Ref : required
 * x2Ref : required
 * gamma : required
 * beta : required
 * biasOptional : optional
 * epsilon : optional
 * additionalOutput : optional
 * x1Ref : required
 * meanOut : required
 * rstdOut : required
 * x2Ref : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplaceAddLayerNormGetWorkspaceSize(
    aclTensor *x1Ref,
    aclTensor *x2Ref,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *biasOptional,
    double epsilon,
    bool additionalOutput,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInplaceAddLayerNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplaceAddLayerNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

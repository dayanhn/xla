
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LAYER_NORM_GRAD_V3_H_
#define ACLNN_LAYER_NORM_GRAD_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLayerNormGradV3GetWorkspaceSize
 * parameters :
 * dy : required
 * x : required
 * rstd : required
 * mean : required
 * gamma : required
 * outputMaskOptional : optional
 * pdXOut : required
 * pdGammaOut : required
 * pdBetaOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormGradV3GetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x,
    const aclTensor *rstd,
    const aclTensor *mean,
    const aclTensor *gamma,
    const aclBoolArray *outputMaskOptional,
    const aclTensor *pdXOut,
    const aclTensor *pdGammaOut,
    const aclTensor *pdBetaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLayerNormGradV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormGradV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

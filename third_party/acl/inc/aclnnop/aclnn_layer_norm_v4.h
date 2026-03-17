
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LAYER_NORM_V4_H_
#define ACLNN_LAYER_NORM_V4_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLayerNormV4GetWorkspaceSize
 * parameters :
 * x : required
 * normalizedShape : required
 * gammaOptional : optional
 * betaOptional : optional
 * epsilon : optional
 * yOut : required
 * meanOut : required
 * rstdOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormV4GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *normalizedShape,
    const aclTensor *gammaOptional,
    const aclTensor *betaOptional,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLayerNormV4
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormV4(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

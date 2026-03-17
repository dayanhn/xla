
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QUANTIZE_ADD_LAYER_NORM_H_
#define ACLNN_QUANTIZE_ADD_LAYER_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQuantizeAddLayerNormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * beta : required
 * bias : required
 * scales : required
 * zeroPointsOptional : optional
 * dtype : required
 * axis : optional
 * epsilon : optional
 * additionalOutput : optional
 * yOut : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQuantizeAddLayerNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *bias,
    const aclTensor *scales,
    const aclTensor *zeroPointsOptional,
    int64_t dtype,
    int64_t axis,
    double epsilon,
    bool additionalOutput,
    const aclTensor *yOut,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQuantizeAddLayerNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQuantizeAddLayerNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

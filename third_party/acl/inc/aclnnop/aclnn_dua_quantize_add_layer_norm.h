
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DUA_QUANTIZE_ADD_LAYER_NORM_H_
#define ACLNN_DUA_QUANTIZE_ADD_LAYER_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDuaQuantizeAddLayerNormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * beta : required
 * bias : required
 * scales1 : required
 * scales2 : required
 * zeroPoints1Optional : optional
 * zeroPoints2Optional : optional
 * dtype : required
 * axis : optional
 * epsilon : optional
 * additionalOutput : optional
 * y1Out : required
 * y2Out : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDuaQuantizeAddLayerNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *bias,
    const aclTensor *scales1,
    const aclTensor *scales2,
    const aclTensor *zeroPoints1Optional,
    const aclTensor *zeroPoints2Optional,
    int64_t dtype,
    int64_t axis,
    double epsilon,
    bool additionalOutput,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDuaQuantizeAddLayerNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDuaQuantizeAddLayerNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

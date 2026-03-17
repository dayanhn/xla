
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_H_
#define ACLNN_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMultiAddRmsNormDynamicQuantGetWorkspaceSize
 * parameters :
 * x1 : dynamic
 * x2 : required
 * gamma : required
 * smoothScale1Optional : optional
 * smoothScale2Optional : optional
 * epsilon : optional
 * y1Out : required
 * y2Out : required
 * xOut : required
 * yOut : required
 * scale1Out : required
 * scale2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *smoothScale1Optional,
    const aclTensor *smoothScale2Optional,
    double epsilon,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *xOut,
    const aclTensor *yOut,
    const aclTensor *scale1Out,
    const aclTensor *scale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMultiAddRmsNormDynamicQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiAddRmsNormDynamicQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

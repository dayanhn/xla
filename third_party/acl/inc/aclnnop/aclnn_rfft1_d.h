
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RFFT1D_H_
#define ACLNN_RFFT1D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRfft1DGetWorkspaceSize
 * parameters :
 * x : required
 * dftOptional : optional
 * n : optional
 * norm : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRfft1DGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *dftOptional,
    int64_t n,
    int64_t norm,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRfft1D
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRfft1D(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

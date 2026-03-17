
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HANS_ENCODE_H_
#define ACLNN_HANS_ENCODE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHansEncodeGetWorkspaceSize
 * parameters :
 * inputTensor : required
 * pdfRef : required
 * statistic : optional
 * reshuff : optional
 * pdfRef : required
 * mantissaOut : required
 * fixedOut : required
 * varOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHansEncodeGetWorkspaceSize(
    const aclTensor *inputTensor,
    aclTensor *pdfRef,
    bool statistic,
    bool reshuff,
    const aclTensor *mantissaOut,
    const aclTensor *fixedOut,
    const aclTensor *varOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHansEncode
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHansEncode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

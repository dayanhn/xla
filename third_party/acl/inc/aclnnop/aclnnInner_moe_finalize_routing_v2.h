
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_FINALIZE_ROUTING_V2_H_
#define ACLNN_INNER_MOE_FINALIZE_ROUTING_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize
 * parameters :
 * expandedX : required
 * expandedRowIdx : required
 * x1Optional : optional
 * x2Optional : optional
 * biasOptional : optional
 * scalesOptional : optional
 * expertIdxOptional : optional
 * dropPadMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor *expandedX,
    const aclTensor *expandedRowIdx,
    const aclTensor *x1Optional,
    const aclTensor *x2Optional,
    const aclTensor *biasOptional,
    const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional,
    int64_t dropPadMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeFinalizeRoutingV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeFinalizeRoutingV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

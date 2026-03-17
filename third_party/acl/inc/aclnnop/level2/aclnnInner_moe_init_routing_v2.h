
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_INIT_ROUTING_V2_H_
#define ACLNN_INNER_MOE_INIT_ROUTING_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeInitRoutingV2GetWorkspaceSize
 * parameters :
 * x : required
 * expertIdx : required
 * activeNum : optional
 * expertCapacity : optional
 * expertNum : optional
 * dropPadMode : optional
 * expertTokensCountOrCumsumFlag : optional
 * expertTokensBeforeCapacityFlag : optional
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expertTokensCountOrCumsumOutOptional : optional
 * expertTokensBeforeCapacityOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional,
    const aclTensor *expertTokensBeforeCapacityOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeInitRoutingV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeInitRoutingV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

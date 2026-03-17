
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_DISTRIBUTE_COMBINE_V2_H_
#define ACLNN_INNER_MOE_DISTRIBUTE_COMBINE_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeDistributeCombineV2GetWorkspaceSize
 * parameters :
 * expandX : required
 * expertIds : required
 * assistInfoForCombine : required
 * epSendCounts : required
 * expertScales : required
 * tpSendCountsOptional : optional
 * xActiveMaskOptional : optional
 * activationScaleOptional : optional
 * weightScaleOptional : optional
 * groupListOptional : optional
 * expandScalesOptional : optional
 * sharedExpertXOptional : optional
 * elasticInfoOptional : optional
 * oriXOptional : optional
 * constExpertAlpha1Optional : optional
 * constExpertAlpha2Optional : optional
 * constExpertVOptional : optional
 * performanceInfoOptional : optional
 * groupEp : required
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * groupTpOptional : optional
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * globalBs : optional
 * outDtype : optional
 * commQuantMode : optional
 * groupListType : optional
 * commAlgOptional : optional
 * zeroExpertNum : optional
 * copyExpertNum : optional
 * constExpertNum : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor *expandX,
    const aclTensor *expertIds,
    const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts,
    const aclTensor *expertScales,
    const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional,
    const aclTensor *groupListOptional,
    const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional,
    const aclTensor *elasticInfoOptional,
    const aclTensor *oriXOptional,
    const aclTensor *constExpertAlpha1Optional,
    const aclTensor *constExpertAlpha2Optional,
    const aclTensor *constExpertVOptional,
    const aclTensor *performanceInfoOptional,
    char *groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    char *groupTpOptional,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t outDtype,
    int64_t commQuantMode,
    int64_t groupListType,
    char *commAlgOptional,
    int64_t zeroExpertNum,
    int64_t copyExpertNum,
    int64_t constExpertNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeDistributeCombineV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeCombineV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_FLASH_ATTENTION_SCORE_GRAD_H_
#define ACLNN_INNER_FLASH_ATTENTION_SCORE_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerFlashAttentionScoreGradGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * dy : required
 * pseShiftOptional : optional
 * dropMaskOptional : optional
 * paddingMaskOptional : optional
 * attenMaskOptional : optional
 * softmaxMaxOptional : optional
 * softmaxSumOptional : optional
 * softmaxInOptional : optional
 * attentionInOptional : optional
 * prefixOptional : optional
 * actualSeqQlenOptional : optional
 * actualSeqKvlenOptional : optional
 * qStartIdxOptional : optional
 * kvStartIdxOptional : optional
 * dScaleQOptional : optional
 * dScaleKOptional : optional
 * dScaleVOptional : optional
 * dScaleDyOptional : optional
 * dScaleOOptional : optional
 * queryRopeOptional : optional
 * keyRopeOptional : optional
 * sinkOptional : optional
 * scaleValue : optional
 * keepProb : optional
 * preTockens : optional
 * nextTockens : optional
 * headNum : required
 * inputLayout : required
 * innerPrecise : optional
 * sparseMode : optional
 * pseType : optional
 * seed : optional
 * offset : optional
 * outDtype : optional
 * softmaxInLayoutOptional : optional
 * dqOut : required
 * dkOut : required
 * dvOut : required
 * dpseOut : required
 * dqRopeOut : required
 * dkRopeOut : required
 * dsinkOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dy,
    const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQlenOptional,
    const aclIntArray *actualSeqKvlenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional,
    const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *sinkOptional,
    double scaleValue,
    double keepProb,
    int64_t preTockens,
    int64_t nextTockens,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,
    int64_t pseType,
    int64_t seed,
    int64_t offset,
    int64_t outDtype,
    char *softmaxInLayoutOptional,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dpseOut,
    const aclTensor *dqRopeOut,
    const aclTensor *dkRopeOut,
    const aclTensor *dsinkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnInnerFlashAttentionScoreGradTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dy,
    const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional,
    const aclTensor *prefixOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKvlenOptional,
    const aclTensor *qStartIdxOptional,
    const aclTensor *kvStartIdxOptional,
    const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional,
    const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *sinkOptional,
    double scaleValue,
    double keepProb,
    int64_t preTockens,
    int64_t nextTockens,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,
    int64_t pseType,
    int64_t seed,
    int64_t offset,
    int64_t outDtype,
    char *softmaxInLayoutOptional,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dpseOut,
    const aclTensor *dqRopeOut,
    const aclTensor *dkRopeOut,
    const aclTensor *dsinkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerFlashAttentionScoreGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFlashAttentionScoreGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

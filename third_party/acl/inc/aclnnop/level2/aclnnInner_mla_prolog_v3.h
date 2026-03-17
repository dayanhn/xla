
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MLA_PROLOG_V3_H_
#define ACLNN_INNER_MLA_PROLOG_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMlaPrologV3GetWorkspaceSize
 * parameters :
 * tokenX : required
 * weightDq : required
 * weightUqQr : required
 * weightUk : required
 * weightDkvKr : required
 * rmsnormGammaCq : required
 * rmsnormGammaCkv : required
 * ropeSin : required
 * ropeCos : required
 * kvCacheRef : required
 * krCacheRef : required
 * cacheIndexOptional : optional
 * dequantScaleXOptional : optional
 * dequantScaleWDqOptional : optional
 * dequantScaleWUqQrOptional : optional
 * dequantScaleWDkvKrOptional : optional
 * quantScaleCkvOptional : optional
 * quantScaleCkrOptional : optional
 * smoothScalesCqOptional : optional
 * actualSeqLenOptional : optional
 * kNopeClipAlphaOptional : optional
 * rmsnormEpsilonCq : optional
 * rmsnormEpsilonCkv : optional
 * cacheModeOptional : optional
 * queryNormFlag : optional
 * weightQuantMode : optional
 * kvCacheQuantMode : optional
 * queryQuantMode : optional
 * ckvkrRepoMode : optional
 * quantScaleRepoMode : optional
 * tileSize : optional
 * qcQrScale : optional
 * kcScale : optional
 * queryOut : required
 * queryRopeOut : required
 * kvCacheRef : required
 * krCacheRef : required
 * dequantScaleQNopeOut : required
 * queryNormOut : required
 * dequantScaleQNormOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMlaPrologV3GetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    aclTensor *kvCacheRef,
    aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv,
    char *cacheModeOptional,
    bool queryNormFlag,
    int64_t weightQuantMode,
    int64_t kvCacheQuantMode,
    int64_t queryQuantMode,
    int64_t ckvkrRepoMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    double qcQrScale,
    double kcScale,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOut,
    const aclTensor *queryNormOut,
    const aclTensor *dequantScaleQNormOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMlaPrologV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMlaPrologV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

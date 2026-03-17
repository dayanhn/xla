
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_RNNV2_H_
#define ACLNN_DYNAMIC_RNNV2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicRNNV2GetWorkspaceSize
 * parameters :
 * x : required
 * weightInput : required
 * weightHidden : required
 * bOptional : optional
 * seqLengthOptional : optional
 * initHOptional : optional
 * initCOptional : optional
 * wciOptional : optional
 * wcfOptional : optional
 * wcoOptional : optional
 * maskOptional : optional
 * cellType : required
 * direction : required
 * cellDepth : required
 * usePeephole : required
 * keepProb : required
 * cellClip : required
 * numProj : required
 * timeMajor : required
 * activation : required
 * recurrentActivation : required
 * forgetBias : required
 * gateOrder : required
 * stateful : required
 * mergeMode : required
 * isTraining : required
 * yOut : required
 * outputHOut : required
 * outputCOut : required
 * iOut : required
 * jOut : required
 * fOut : required
 * oOut : required
 * tanhcOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicRNNV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightInput,
    const aclTensor *weightHidden,
    const aclTensor *bOptional,
    const aclTensor *seqLengthOptional,
    const aclTensor *initHOptional,
    const aclTensor *initCOptional,
    const aclTensor *wciOptional,
    const aclTensor *wcfOptional,
    const aclTensor *wcoOptional,
    const aclTensor *maskOptional,
    char *cellType,
    char *direction,
    int64_t cellDepth,
    bool usePeephole,
    double keepProb,
    double cellClip,
    int64_t numProj,
    bool timeMajor,
    char *activation,
    char *recurrentActivation,
    double forgetBias,
    char *gateOrder,
    bool stateful,
    char *mergeMode,
    bool isTraining,
    const aclTensor *yOut,
    const aclTensor *outputHOut,
    const aclTensor *outputCOut,
    const aclTensor *iOut,
    const aclTensor *jOut,
    const aclTensor *fOut,
    const aclTensor *oOut,
    const aclTensor *tanhcOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicRNNV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicRNNV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

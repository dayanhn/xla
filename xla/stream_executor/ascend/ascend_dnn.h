/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_ASCEND_ASCEND_DNN_H_ 
#define XLA_SERVICE_ASCEND_ASCEND_DNN_H_

#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace ascend {

// Forward declaration
class AscendAccess;

// Ascend DNN support implementation
class AscendSupport : public dnn::DnnSupport {
 public:
  explicit AscendSupport(StreamExecutor* parent);
  ~AscendSupport() override;

  // Initialize Ascend DNN support
  absl::Status Init() override;

  // Notify stream destroyed
  void NotifyStreamDestroyed(Stream* stream) override;

  // Get version info
  absl::StatusOr<dnn::VersionInfo> GetVersion() override;

  // Convolution operations
  absl::Status DoConvolveForward(Stream* stream,
                                 const dnn::BatchDescriptor& input_descriptor,
                                 const DeviceAddressBase& input_data,
                                 const dnn::FilterDescriptor& filter_descriptor,
                                 const DeviceAddressBase& filter_data,
                                 const dnn::ConvolutionDescriptor& convolution_descriptor,
                                 const dnn::BatchDescriptor& output_descriptor,
                                 DeviceAddressBase* output_data,
                                 const dnn::AlgorithmConfig& algorithm_config,
                                 ScratchAllocator* workspace_allocator,
                                 dnn::ProfileResult* output_profile_result) override;

  absl::Status DoConvolveBackwardData(Stream* stream,
                                      const dnn::FilterDescriptor& filter_descriptor,
                                      const DeviceAddressBase& filter_data,
                                      const dnn::BatchDescriptor& output_backprop_descriptor,
                                      const DeviceAddressBase& output_backprop_data,
                                      const dnn::ConvolutionDescriptor& convolution_descriptor,
                                      const dnn::BatchDescriptor& input_backprop_descriptor,
                                      DeviceAddressBase* input_backprop_data,
                                      const dnn::AlgorithmConfig& algorithm_config,
                                      ScratchAllocator* workspace_allocator,
                                      dnn::ProfileResult* output_profile_result) override;

  absl::Status DoConvolveBackwardFilter(Stream* stream,
                                        const dnn::BatchDescriptor& input_descriptor,
                                        const DeviceAddressBase& input_data,
                                        const dnn::BatchDescriptor& output_backprop_descriptor,
                                        const DeviceAddressBase& output_backprop_data,
                                        const dnn::ConvolutionDescriptor& convolution_descriptor,
                                        const dnn::FilterDescriptor& filter_backprop_descriptor,
                                        DeviceAddressBase* filter_backprop_data,
                                        const dnn::AlgorithmConfig& algorithm_config,
                                        ScratchAllocator* workspace_allocator,
                                        dnn::ProfileResult* output_profile_result) override;

  // Pooling operations
  absl::Status DoPoolForward(Stream* stream,
                            const dnn::BatchDescriptor& input_descriptor,
                            const DeviceAddressBase& input_data,
                            const dnn::PoolingDescriptor& pooling_descriptor,
                            const dnn::BatchDescriptor& output_descriptor,
                            DeviceAddressBase* output_data,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) override;

  absl::Status DoPoolBackward(Stream* stream,
                             const dnn::BatchDescriptor& input_descriptor,
                             const DeviceAddressBase& input_data,
                             const dnn::BatchDescriptor& output_descriptor,
                             const DeviceAddressBase& output_data,
                             const dnn::BatchDescriptor& output_backprop_descriptor,
                             const DeviceAddressBase& output_backprop_data,
                             const dnn::PoolingDescriptor& pooling_descriptor,
                             const dnn::BatchDescriptor& input_backprop_descriptor,
                             DeviceAddressBase* input_backprop_data,
                             ScratchAllocator* workspace_allocator,
                             dnn::ProfileResult* output_profile_result) override;

  // Batch normalization operations
  absl::Status DoBatchNormForwardTraining(Stream* stream,
                                         dnn::BatchNormMode mode,
                                         const dnn::BatchDescriptor& input_descriptor,
                                         const DeviceAddressBase& input_data,
                                         const dnn::BatchDescriptor& scale_offset_descriptor,
                                         const DeviceAddressBase& scale_data,
                                         const DeviceAddressBase& offset_data,
                                         const dnn::BatchDescriptor& output_descriptor,
                                         DeviceAddressBase* output_data,
                                         const dnn::BatchDescriptor& mean_descriptor,
                                         DeviceAddressBase* mean_data,
                                         const dnn::BatchDescriptor& variance_descriptor,
                                         DeviceAddressBase* variance_data,
                                         double epsilon,
                                         dnn::ActivationMode activation_mode,
                                         double activation_max_value,
                                         ScratchAllocator* workspace_allocator,
                                         dnn::ProfileResult* output_profile_result) override;

  absl::Status DoBatchNormBackward(Stream* stream,
                                  dnn::BatchNormMode mode,
                                  const dnn::BatchDescriptor& input_descriptor,
                                  const DeviceAddressBase& input_data,
                                  const dnn::BatchDescriptor& output_backprop_descriptor,
                                  const DeviceAddressBase& output_backprop_data,
                                  const dnn::BatchDescriptor& scale_offset_descriptor,
                                  const DeviceAddressBase& scale_data,
                                  const DeviceAddressBase& mean_data,
                                  const DeviceAddressBase& variance_data,
                                  double epsilon,
                                  dnn::ActivationMode activation_mode,
                                  double activation_max_value,
                                  const dnn::BatchDescriptor& input_backprop_descriptor,
                                  DeviceAddressBase* input_backprop_data,
                                  const dnn::BatchDescriptor& scale_offset_backprop_descriptor,
                                  DeviceAddressBase* scale_backprop_data,
                                  DeviceAddressBase* offset_backprop_data,
                                  ScratchAllocator* workspace_allocator,
                                  dnn::ProfileResult* output_profile_result) override;

  // LRN operations
  absl::Status DoLrnForward(Stream* stream,
                            const dnn::BatchDescriptor& input_descriptor,
                            const DeviceAddressBase& input_data,
                            const dnn::NormalizeDescriptor& normalize_descriptor,
                            const dnn::BatchDescriptor& output_descriptor,
                            DeviceAddressBase* output_data,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) override;

  absl::Status DoLrnBackward(Stream* stream,
                             const dnn::BatchDescriptor& input_descriptor,
                             const DeviceAddressBase& input_data,
                             const dnn::BatchDescriptor& output_descriptor,
                             const DeviceAddressBase& output_data,
                             const dnn::BatchDescriptor& output_backprop_descriptor,
                             const DeviceAddressBase& output_backprop_data,
                             const dnn::NormalizeDescriptor& normalize_descriptor,
                             const dnn::BatchDescriptor& input_backprop_descriptor,
                             DeviceAddressBase* input_backprop_data,
                             ScratchAllocator* workspace_allocator,
                             dnn::ProfileResult* output_profile_result) override;

  // RNN operations
  absl::Status DoRnnForward(Stream* stream,
                           const dnn::RnnDescriptor& rnn_desc,
                           const dnn::RnnSequenceTensorDescriptor& input_desc,
                           const DeviceAddressBase& input_data,
                           const DeviceAddress<int>& seq_lengths_data,
                           const dnn::RnnStateTensorDescriptor& input_h_desc,
                           const DeviceAddressBase& input_h_data,
                           const dnn::RnnStateTensorDescriptor& input_c_desc,
                           const DeviceAddressBase& input_c_data,
                           const DeviceAddressBase& params,
                           const dnn::RnnSequenceTensorDescriptor& output_desc,
                           DeviceAddressBase* output_data,
                           const dnn::RnnStateTensorDescriptor& output_h_desc,
                           DeviceAddressBase* output_h_data,
                           const dnn::RnnStateTensorDescriptor& output_c_desc,
                           DeviceAddressBase* output_c_data,
                           bool is_training,
                           ScratchAllocator* reserve_space_allocator,
                           ScratchAllocator* workspace_allocator,
                           dnn::ProfileResult* output_profile_result) override;

  absl::Status DoRnnBackward(Stream* stream,
                            const dnn::RnnDescriptor& rnn_desc,
                            const dnn::RnnSequenceTensorDescriptor& input_desc,
                            const DeviceAddressBase& input_data,
                            const DeviceAddress<int>& seq_lengths_data,
                            const dnn::RnnStateTensorDescriptor& input_h_desc,
                            const DeviceAddressBase& input_h_data,
                            const dnn::RnnStateTensorDescriptor& input_c_desc,
                            const DeviceAddressBase& input_c_data,
                            const DeviceAddressBase& params,
                            const dnn::RnnSequenceTensorDescriptor& output_desc,
                            const DeviceAddressBase& output_data,
                            const dnn::RnnStateTensorDescriptor& output_h_desc,
                            const DeviceAddressBase& output_h_data,
                            const dnn::RnnStateTensorDescriptor& output_c_desc,
                            const DeviceAddressBase& output_c_data,
                            const DeviceAddressBase& output_backprop_data,
                            const DeviceAddressBase& output_h_backprop_data,
                            const DeviceAddressBase& output_c_backprop_data,
                            DeviceAddressBase* input_backprop_data,
                            DeviceAddressBase* input_h_backprop_data,
                            DeviceAddressBase* input_c_backprop_data,
                            DeviceAddressBase* params_backprop_data,
                            DeviceAddress<uint8_t>* reserve_space_data,
                            ScratchAllocator* workspace_allocator,
                            dnn::ProfileResult* output_profile_result) override;

  // CTC loss operations
  absl::Status DoCtcLoss(Stream* stream,
                        const dnn::BatchDescriptor& probs_descriptor,
                        const DeviceAddressBase& probs_data,
                        absl::Span<const int> labels_data,
                        absl::Span<const int> labels_lengths_data,
                        absl::Span<const int> input_lengths_data,
                        DeviceAddressBase& costs_data,
                        const dnn::BatchDescriptor& grads_descriptor,
                        DeviceAddressBase* grads_data,
                        int ctc_loss_algo_id,
                        ScratchAllocator* workspace_allocator) override;

  // Descriptor creation
  absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> CreateRnnDescriptor(
      int num_layers, int hidden_size, int input_size, int cell_size,
      int batch_size, dnn::RnnInputMode input_mode,
      dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
      dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
      const EngineOptions& engine_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                   int data_size, dnn::DataType data_type)
      override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                   int data_size,
                                   const absl::Span<const int>& seq_lengths,
                                   bool time_major, dnn::DataType data_type)
      override;

  absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  CreateRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                dnn::DataType data_type) override;

  // Algorithm retrieval
  absl::StatusOr<std::vector<dnn::AlgorithmDesc>> GetConvolveAlgorithms(
      ConvolutionKind kind,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor) override;

  // Workspace size calculation
  absl::StatusOr<size_t> GetConvolveWorkspaceSize(
      ConvolutionKind kind,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::AlgorithmDesc& algorithm) override;

  absl::StatusOr<size_t> GetPoolingWorkspaceSize(
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::PoolingDescriptor& pooling_descriptor,
      const dnn::BatchDescriptor& output_descriptor) override;

  absl::StatusOr<size_t> GetBatchNormWorkspaceSize(
      dnn::BatchNormMode mode,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::BatchDescriptor& scale_offset_descriptor,
      dnn::ActivationMode activation_mode) override;

  absl::StatusOr<size_t> GetLrnWorkspaceSize(
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::NormalizeDescriptor& normalize_descriptor) override;

  absl::StatusOr<size_t> GetRnnWorkspaceSize(
      const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      bool is_training) override;

  absl::StatusOr<size_t> GetRnnReserveSpaceSize(
      const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc) override;

  absl::StatusOr<size_t> GetCtcLossWorkspaceSize(
      const dnn::BatchDescriptor& probs_descriptor,
      int ctc_loss_algo_id) override;

 private:
  // Internal implementations
  absl::Status DoConvolveForwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceAddressBase& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase* output_data,
      const dnn::AlgorithmConfig& algorithm_config,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoConvolveBackwardDataImpl(
      Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
      const DeviceAddressBase& filter_data,
      const dnn::BatchDescriptor& output_backprop_descriptor,
      const DeviceAddressBase& output_backprop_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_backprop_descriptor,
      DeviceAddressBase* input_backprop_data,
      const dnn::AlgorithmConfig& algorithm_config,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoConvolveBackwardFilterImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::BatchDescriptor& output_backprop_descriptor,
      const DeviceAddressBase& output_backprop_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_backprop_descriptor,
      DeviceAddressBase* filter_backprop_data,
      const dnn::AlgorithmConfig& algorithm_config,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoPoolForwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::PoolingDescriptor& pooling_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase* output_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoPoolBackwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      const DeviceAddressBase& output_data,
      const dnn::BatchDescriptor& output_backprop_descriptor,
      const DeviceAddressBase& output_backprop_data,
      const dnn::PoolingDescriptor& pooling_descriptor,
      const dnn::BatchDescriptor& input_backprop_descriptor,
      DeviceAddressBase* input_backprop_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoBatchNormForwardTrainingImpl(
      Stream* stream, dnn::BatchNormMode mode,
      const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::BatchDescriptor& scale_offset_descriptor,
      const DeviceAddressBase& scale_data,
      const DeviceAddressBase& offset_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase* output_data,
      const dnn::BatchDescriptor& mean_descriptor,
      DeviceAddressBase* mean_data,
      const dnn::BatchDescriptor& variance_descriptor,
      DeviceAddressBase* variance_data,
      double epsilon,
      dnn::ActivationMode activation_mode,
      double activation_max_value,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoBatchNormBackwardImpl(
      Stream* stream, dnn::BatchNormMode mode,
      const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::BatchDescriptor& output_backprop_descriptor,
      const DeviceAddressBase& output_backprop_data,
      const dnn::BatchDescriptor& scale_offset_descriptor,
      const DeviceAddressBase& scale_data,
      const DeviceAddressBase& mean_data,
      const DeviceAddressBase& variance_data,
      double epsilon,
      dnn::ActivationMode activation_mode,
      double activation_max_value,
      const dnn::BatchDescriptor& input_backprop_descriptor,
      DeviceAddressBase* input_backprop_data,
      const dnn::BatchDescriptor& scale_offset_backprop_descriptor,
      DeviceAddressBase* scale_backprop_data,
      DeviceAddressBase* offset_backprop_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoLrnForwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase* output_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoLrnBackwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      const DeviceAddressBase& output_data,
      const dnn::BatchDescriptor& output_backprop_descriptor,
      const DeviceAddressBase& output_backprop_data,
      const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& input_backprop_descriptor,
      DeviceAddressBase* input_backprop_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoRnnForwardImpl(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceAddressBase& input_data,
      const DeviceAddress<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceAddressBase& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceAddressBase& input_c_data,
      const DeviceAddressBase& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      DeviceAddressBase* output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      DeviceAddressBase* output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      DeviceAddressBase* output_c_data,
      bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoRnnBackwardImpl(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceAddressBase& input_data,
      const DeviceAddress<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceAddressBase& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceAddressBase& input_c_data,
      const DeviceAddressBase& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      const DeviceAddressBase& output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      const DeviceAddressBase& output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      const DeviceAddressBase& output_c_data,
      const DeviceAddressBase& output_backprop_data,
      const DeviceAddressBase& output_h_backprop_data,
      const DeviceAddressBase& output_c_backprop_data,
      DeviceAddressBase* input_backprop_data,
      DeviceAddressBase* input_h_backprop_data,
      DeviceAddressBase* input_c_backprop_data,
      DeviceAddressBase* params_backprop_data,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoCtcLossImpl(
      Stream* stream, const dnn::BatchDescriptor& probs_descriptor,
      const DeviceAddressBase& probs_data,
      absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data,
      DeviceAddressBase& costs_data,
      const dnn::BatchDescriptor& grads_descriptor,
      DeviceAddressBase* grads_data,
      int ctc_loss_algo_id,
      ScratchAllocator* workspace_allocator);

  // StreamExecutor parent
  StreamExecutor* parent_;

  // Ascend access
  std::unique_ptr<AscendAccess> ascend_;
};

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_SERVICE_ASCEND_ASCEND_DNN_H_

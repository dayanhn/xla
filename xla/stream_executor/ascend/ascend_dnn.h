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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_DNN_H_ 
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_DNN_H_

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

  // Pooling operations
  absl::Status DoPoolForward(
      dnn::DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const dnn::BatchDescriptor& input_dimensions, DeviceAddressBase input_data,
      const dnn::BatchDescriptor& output_dimensions, DeviceAddressBase output_data,
      ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolForward(
      dnn::DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const EngineOptions& engine_options,
      const dnn::BatchDescriptor& input_dimensions, DeviceAddressBase input_data,
      const dnn::BatchDescriptor& output_dimensions, DeviceAddressBase output_data,
      ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(
      dnn::DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const dnn::BatchDescriptor& input_dimensions, DeviceAddressBase input_data,
      const dnn::BatchDescriptor& output_dimensions, DeviceAddressBase output_data,
      DeviceAddressBase input_diff_data, DeviceAddressBase output_diff_data,
      ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(
      dnn::DataType element_type, Stream* stream,
      const dnn::PoolingDescriptor& pooling_dimensions,
      const EngineOptions& engine_options,
      const dnn::BatchDescriptor& input_dimensions, DeviceAddressBase input_data,
      const dnn::BatchDescriptor& output_dimensions, DeviceAddressBase output_data,
      DeviceAddressBase input_diff_data, DeviceAddressBase output_diff_data,
      ScratchAllocator* workspace_allocator) override;

  // Batch normalization operations
  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceAddress<float>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& estimated_mean,
      const DeviceAddress<float>& estimated_variance,
      const DeviceAddress<float>& side_input, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor, dnn::ActivationMode activation_mode,
      DeviceAddress<float>* y, DeviceAddress<float>* batch_mean,
      DeviceAddress<float>* batch_var, DeviceAddress<float>* reserve_space_1,
      DeviceAddress<float>* reserve_space_2, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceAddress<Eigen::half>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& estimated_mean,
      const DeviceAddress<float>& estimated_variance,
      const DeviceAddress<Eigen::half>& side_input,
      const dnn::BatchDescriptor& x_desc, const dnn::BatchDescriptor& scale_offset_desc,
      const double epsilon, const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceAddress<Eigen::half>* y,
      DeviceAddress<float>* batch_mean, DeviceAddress<float>* batch_var,
      DeviceAddress<float>* reserve_space_1, DeviceAddress<float>* reserve_space_2,
      bool is_training, ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceAddress<float>& y_backprop,
      const DeviceAddress<float>& x, const DeviceAddress<float>& scale,
      const DeviceAddress<float>& offset, const DeviceAddress<float>& mean,
      const DeviceAddress<float>& inv_var, const DeviceAddress<float>& y,
      const dnn::BatchDescriptor& x_desc, const dnn::BatchDescriptor& scale_offset_desc,
      const double epsilon, dnn::ActivationMode activation_mode,
      DeviceAddress<float>* x_backprop, DeviceAddress<float>* scale_backprop,
      DeviceAddress<float>* offset_backprop,
      DeviceAddress<float>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceAddress<Eigen::half>& y_backprop,
      const DeviceAddress<Eigen::half>& x, const DeviceAddress<float>& scale,
      const DeviceAddress<float>& offset, const DeviceAddress<float>& mean,
      const DeviceAddress<float>& inv_var, const DeviceAddress<Eigen::half>& y,
      const dnn::BatchDescriptor& x_desc, const dnn::BatchDescriptor& scale_offset_desc,
      const double epsilon, dnn::ActivationMode activation_mode,
      DeviceAddress<Eigen::half>* x_backprop,
      DeviceAddress<float>* scale_backprop, DeviceAddress<float>* offset_backprop,
      DeviceAddress<Eigen::half>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  // RNN operations
  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<Eigen::half>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<Eigen::half>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<Eigen::half>& input_c_data,
                    const DeviceAddress<Eigen::half>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<Eigen::half>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<Eigen::half>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<Eigen::half>* output_c_data,
                    bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<float>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<float>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<float>& input_c_data,
                    const DeviceAddress<float>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<float>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<float>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<float>* output_c_data,
                    bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<double>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<double>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<double>& input_c_data,
                    const DeviceAddress<double>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<double>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<double>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<double>* output_c_data,
                    bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(
      Stream* stream, const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      const DeviceAddress<Eigen::half>& input_data,
      const DeviceAddress<int>& seq_lengths_data,
      const dnn::RnnStateTensorDescriptor& input_h_desc,
      const DeviceAddress<Eigen::half>& input_h_data,
      const dnn::RnnStateTensorDescriptor& input_c_desc,
      const DeviceAddress<Eigen::half>& input_c_data,
      const DeviceAddress<Eigen::half>& params,
      const dnn::RnnSequenceTensorDescriptor& output_desc,
      const DeviceAddress<Eigen::half>& output_data,
      const dnn::RnnStateTensorDescriptor& output_h_desc,
      const DeviceAddress<Eigen::half>& output_h_data,
      const dnn::RnnStateTensorDescriptor& output_c_desc,
      const DeviceAddress<Eigen::half>& output_c_data,
      const DeviceAddress<Eigen::half>& output_backprop_data,
      const DeviceAddress<Eigen::half>& output_h_backprop_data,
      const DeviceAddress<Eigen::half>& output_c_backprop_data,
      DeviceAddress<Eigen::half>* input_backprop_data,
      DeviceAddress<Eigen::half>* input_h_backprop_data,
      DeviceAddress<Eigen::half>* input_c_backprop_data,
      DeviceAddress<Eigen::half>* params_backprop_data,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceAddress<float>& input_data,
                     const DeviceAddress<int>& seq_lengths_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceAddress<float>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceAddress<float>& input_c_data,
                     const DeviceAddress<float>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceAddress<float>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceAddress<float>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceAddress<float>& output_c_data,
                     const DeviceAddress<float>& output_backprop_data,
                     const DeviceAddress<float>& output_h_backprop_data,
                     const DeviceAddress<float>& output_c_backprop_data,
                     DeviceAddress<float>* input_backprop_data,
                     DeviceAddress<float>* input_h_backprop_data,
                     DeviceAddress<float>* input_c_backprop_data,
                     DeviceAddress<float>* params_backprop_data,
                     DeviceAddress<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceAddress<double>& input_data,
                     const DeviceAddress<int>& seq_lengths_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceAddress<double>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceAddress<double>& input_c_data,
                     const DeviceAddress<double>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceAddress<double>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceAddress<double>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceAddress<double>& output_c_data,
                     const DeviceAddress<double>& output_backprop_data,
                     const DeviceAddress<double>& output_h_backprop_data,
                     const DeviceAddress<double>& output_c_backprop_data,
                     DeviceAddress<double>* input_backprop_data,
                     DeviceAddress<double>* input_h_backprop_data,
                     DeviceAddress<double>* input_c_backprop_data,
                     DeviceAddress<double>* params_backprop_data,
                     DeviceAddress<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

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

  // Get convolution algorithms
  absl::StatusOr<std::vector<dnn::AlgorithmDesc>> GetConvolveAlgorithms(
      dnn::ConvolutionKind kind,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor);

  // Get convolution workspace size
  absl::StatusOr<size_t> GetConvolveWorkspaceSize(
      dnn::ConvolutionKind kind,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::AlgorithmDesc& algorithm);

  // Get pooling workspace size
  absl::StatusOr<size_t> GetPoolingWorkspaceSize(
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::PoolingDescriptor& pooling_descriptor,
      const dnn::BatchDescriptor& output_descriptor);

  // Get LRN workspace size
  absl::StatusOr<size_t> GetLrnWorkspaceSize(
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::NormalizeDescriptor& normalize_descriptor);

  // Get RNN workspace size
  absl::StatusOr<size_t> GetRnnWorkspaceSize(
      const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc,
      bool is_training);

  // Get RNN reserve space size
  absl::StatusOr<size_t> GetRnnReserveSpaceSize(
      const dnn::RnnDescriptor& rnn_desc,
      const dnn::RnnSequenceTensorDescriptor& input_desc);

  // Get CTC loss workspace size
  absl::StatusOr<size_t> GetCtcLossWorkspaceSize(
      const dnn::BatchDescriptor& probs_descriptor,
      int ctc_loss_algo_id);

 private:
  // Internal implementations
  absl::Status DoPoolForwardImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceAddressBase& input_data,
      const dnn::PoolingDescriptor& pooling_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase* output_data,
      ScratchAllocator* workspace_allocator);

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
      ScratchAllocator* workspace_allocator);

  // StreamExecutor parent
  StreamExecutor* parent_;

  // Ascend access
  std::unique_ptr<AscendAccess> ascend_;
};

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_DNN_H_

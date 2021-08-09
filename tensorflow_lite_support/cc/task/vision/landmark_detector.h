#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_LANDMARK_DETECTOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_LANDMARK_DETECTOR_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/kernels/register.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/vision/core/base_vision_task_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmarks_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmark_detector_options_proto_inc.h"

namespace tflite {
namespace task {
namespace vision {


class LandmarkDetector : public BaseVisionTaskApi<LandmarkResult> {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageClassifier from the provided options. A non-default
  // OpResolver can be specified in order to support custom Ops or specify a
  // subset of built-in Ops.
  static tflite::support::StatusOr<std::unique_ptr<LandmarkDetector>>
  CreateFromOptions(
      const LandmarkDetectorOptions& options,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite_shims::ops::builtin::BuiltinOpResolver>());


  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer);


  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer, const BoundingBox& roi);

 protected:
  // The options used to build this ImageClassifier.
  std::unique_ptr<LandmarkDetectorOptions> options_;

  // The list of classification heads associated with the corresponding output
  // tensors. Built from TFLite Model Metadata.
  // c1 no classification heads
  // std::vector<ClassificationHead> classification_heads_;

  // Post-processing to transform the raw model outputs into classification
  // results.
  tflite::support::StatusOr<LandmarkResult> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const FrameBuffer& frame_buffer, const BoundingBox& roi) override;

  // Performs sanity checks on the provided ImageClassifierOptions.
  static absl::Status SanityCheckOptions(const LandmarkDetectorOptions& options);

  // Initializes the ImageClassifier from the provided ImageClassifierOptions,
  // whose ownership is transferred to this object.
  absl::Status Init(std::unique_ptr<LandmarkDetectorOptions> options);

  // Performs pre-initialization actions.
  virtual absl::Status PreInit();
  // Performs post-initialization actions.
  // virtual absl::Status PostInit();

 private:
  // Performs sanity checks on the model outputs and extracts their metadata.
  // c2 disabling 
  // absl::Status CheckAndSetOutputs();

  // Performs sanity checks on the class whitelist/blacklist and forms the class


  // Given a ClassificationResult object containing class indices, fills the
  // name and display name from the label map(s).


  // The number of output tensors. This corresponds to the number of
  // classification heads.
  int num_outputs_;
  // Whether the model features quantized inference type (QUANTIZED_UINT8). This
  // is currently detected by checking if all output tensors data type is uint8.
  bool has_uint8_outputs_;



  // List of score calibration parameters, if any. Built from TFLite Model
  // Metadata.
  // c3 no score calliberation needed
  // std::vector<std::unique_ptr<ScoreCalibration>> score_calibrations_;
};

}  // namespace vision
}  // namespace task
}  // namespace tflite

#endif
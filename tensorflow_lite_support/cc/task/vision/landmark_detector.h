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


  static tflite::support::StatusOr<std::unique_ptr<LandmarkDetector>>
  CreateFromOptions(
      const LandmarkDetectorOptions& options
      );


  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer);


  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer, const BoundingBox& roi);

 protected:
  // The options used to build this LandmarkDetector.
  std::unique_ptr<LandmarkDetectorOptions> options_;

  // Post-processing to transform the raw model outputs into landmarks
  // results.
  tflite::support::StatusOr<LandmarkResult> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const FrameBuffer& frame_buffer, const BoundingBox& roi) override;

  // Performs sanity checks on the provided LandmarkDetectorOptions.
  static absl::Status SanityCheckOptions(const LandmarkDetectorOptions& options);

  // Initializes the LandmarkDetector from the provided LandmarkDetectorOptions,
  // whose ownership is transferred to this object.
  absl::Status Init(std::unique_ptr<LandmarkDetectorOptions> options);

  // Performs pre-initialization actions.
  virtual absl::Status PreInit();


};

}  // namespace vision
}  // namespace task
}  // namespace tflite

#endif
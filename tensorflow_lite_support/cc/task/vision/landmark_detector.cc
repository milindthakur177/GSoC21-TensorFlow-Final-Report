#include "tensorflow_lite_support/cc/task/vision/landmark_detector.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/core/label_map_item.h"
#include "tensorflow_lite_support/cc/task/vision/proto/class_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace task {
namespace vision {

namespace {

using ::absl::StatusCode;
using ::tflite::metadata::ModelMetadataExtractor;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::AssertAndReturnTypedTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;



}  // namespace

// Defining Number of keypoints
int numKeyPoints = 17;
/* static */
StatusOr<std::unique_ptr<LandmarkDetector>> LandmarkDetector::CreateFromOptions(
    const LandmarkDetectorOptions& options) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile outlives the constructed object.
  auto options_copy = absl::make_unique<LandmarkDetectorOptions>(options);

  std::unique_ptr<LandmarkDetector> landmark_detector;
  if (options_copy->has_model_file_with_metadata()) {
    ASSIGN_OR_RETURN(
        landmark_detector,
        TaskAPIFactory::CreateFromExternalFileProto<LandmarkDetector>(
            &options_copy->model_file_with_metadata()
            ));
  } else if (options_copy->base_options().has_model_file()) {
    ASSIGN_OR_RETURN(landmark_detector,
                     TaskAPIFactory::CreateFromBaseOptions<LandmarkDetector>(
                         &options_copy->base_options(), std::move(resolver)));
  } else {
    // Should never happen because of SanityCheckOptions.
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."),
        TfLiteSupportStatus::kInvalidArgumentError);
  }

  RETURN_IF_ERROR(landmark_detector->Init(std::move(options_copy)));

  return landmark_detector;
}

/* static */
absl::Status LandmarkDetector::SanityCheckOptions(
    const LandmarkDetectorOptions& options) {
  int num_input_models = (options.base_options().has_model_file() ? 1 : 0) +
                         (options.has_model_file_with_metadata() ? 1 : 0);
  if (num_input_models != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found %d.",
                        num_input_models),
        TfLiteSupportStatus::kInvalidArgumentError);
  }
 
  return absl::OkStatus();
}

absl::Status LandmarkDetector::Init(
    std::unique_ptr<LandmarkDetectorOptions> options) {
  // Set options.
  options_ = std::move(options);

  // Perform pre-initialization actions (by default, sets the process engine for
  // image pre-processing to kLibyuv as a sane default).
  RETURN_IF_ERROR(PreInit());

  // Sanity check and set inputs and outputs.
  RETURN_IF_ERROR(CheckAndSetInputs());


  return absl::OkStatus();
}

absl::Status LandmarkDetector::PreInit() {
  SetProcessEngine(FrameBufferUtils::ProcessEngine::kLibyuv);
  return absl::OkStatus();
}

StatusOr<LandmarkResult> LandmarkDetector::Detect(
    const FrameBuffer& frame_buffer) {
  BoundingBox roi;
  roi.set_width(frame_buffer.dimension().width);
  roi.set_height(frame_buffer.dimension().height);
  return Detect(frame_buffer, roi);
}

StatusOr<LandmarkResult> LandmarkDetector::Detect(
    const FrameBuffer& frame_buffer, const BoundingBox& roi) {
  return InferWithFallback(frame_buffer, roi);
}

StatusOr<LandmarkResult> LandmarkDetector::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const FrameBuffer& /*frame_buffer*/, const BoundingBox& /*roi*/) {
  if (output_tensors.size() != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Expected 1 output tensors, found %d",
                        output_tensors.size()));
  }

  const TfLiteTensor* output_tensor = output_tensors[0];
  const float* tensor_output = AssertAndReturnTypedTensor<float>(output_tensor);
	
  LandmarkResult result;

	for(int i =0 ; i<numKeyPoints ; ++i){

    Landmark* landmarks = result.add_landmarks();

		landmarks->set_key_y(tensor_output[3*i+0]) ;
		landmarks->set_key_x(tensor_output[3*i+1]) ;
		landmarks->set_score(tensor_output[3*i+2]);

  }
  return result;
}


} 
}  
}
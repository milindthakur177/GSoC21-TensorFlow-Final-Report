#include "tensorflow_lite_support/cc/task/vision/landmark_detector.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmarks_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmark_detector_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace task {
namespace vision {
namespace {

using ::testing::HasSubstr;
using ::testing::Optional;
using ::tflite::support::kTfLiteSupportPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::JoinPath;
using ::tflite::task::core::PopulateTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;

int num_keypoints = 17;
float total_score=0.0;
float avg_score;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";

constexpr char kMobileNetFloatWithMetadata[] =
    "movenet_lightening_with_metadata.tflite";

std::vector<float> KEY_Y = {0.31545776, 0.29907033, 0.3031672, 0.3031672, 0.30726406,0.3482326, 0.4096854, 0.30726406, 0.4260728, 
                                    0.2581018, 0.4260728, 0.49162248, 0.5530753, 0.34413573, 0.73333687, 0.27858606, 0.9299859};

std::vector<float> KEY_X = {0.4260728, 0.44246024, 0.44655707, 0.48752564, 0.47523507, 0.589947 ,0.48342878,0.72514313, 0.34413573,
                                    0.8357582, 0.24581124,0.73743373, 0.6841746, 0.88492055, 0.7210463, 0.8644362, 0.7128526};

std::vector<float> SCORE = {0.70, 0.635, 0.245, 0.88, 0.753, 0.753, 0.90, 0.925, 0.88, 0.753, 0.80, 0.80, 0.843, 0.80, 0.966, 0.635, 0.942};

float AVG_SCORE = 0.776;

class DetectTest : public tflite_shims::testing::Test {};

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

TEST_F(DetectTest, SucceedsWithFloatModel) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("girl.jpg"));
  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      rgb_image.pixel_data,
      FrameBuffer::Dimension{rgb_image.width, rgb_image.height});

  LandmarkDetectorOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath( "./" /*test src dir*/,kTestDataDirectory,
               kMobileNetFloatWithMetadata));
  SUPPORT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LandmarkDetector> landmark_detector,
                       LandmarkDetector::CreateFromOptions(options));
  
  StatusOr<LandmarkResult> result_or =
      landmark_detector->Detect(*frame_buffer);
  ImageDataFree(&rgb_image);
  SUPPORT_ASSERT_OK(result_or);

  const LandmarkResult& result = result_or.value();
  
  for (int i =0 ; i<num_keypoints ; ++i){
    EXPECT_NEAR(result.landmarks(i).key_y(), KEY_Y[i], 0.1);
    EXPECT_NEAR(result.landmarks(i).key_x(), KEY_X[i], 0.1);
    EXPECT_NEAR(result.landmarks(i).score(), SCORE[i], 0.1);
    total_score = total_score +result.landmarks(i).score();
  }

  avg_score = total_score/17;
  EXPECT_NEAR(avg_score, AVG_SCORE, 0.1);
  

}

class CreateFromOptionsTest : public tflite_shims::testing::Test {};


TEST_F(CreateFromOptionsTest, FailsWithTwoModelSources) {
  LandmarkDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatWithMetadata));
  options.mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatWithMetadata));

  StatusOr<std::unique_ptr<LandmarkDetector>> landmark_detector_or =
      LandmarkDetector::CreateFromOptions(options);

  EXPECT_EQ(landmark_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(landmark_detector_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 2."));
  EXPECT_THAT(landmark_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  LandmarkDetectorOptions options;

  StatusOr<std::unique_ptr<LandmarkDetector>> landmark_detector_or =
      LandmarkDetector::CreateFromOptions(options);

  EXPECT_EQ(landmark_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(landmark_detector_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."));
  EXPECT_THAT(landmark_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}


}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
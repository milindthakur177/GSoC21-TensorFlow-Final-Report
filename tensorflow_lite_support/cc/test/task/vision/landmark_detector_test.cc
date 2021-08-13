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

int numKeyPoints = 17;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";

constexpr char kMobileNetFloatWithMetadata[] =
    "lite-model_movenet_singlepose_lightning_tflite_int8_4_with_metadata.tflite";

std::vector<float> key_y_golden = {0.1442298, 0.1300826, 0.13248327, 0.14726797, 0.15074761, 0.2509237, 0.25660858, 0.34447512, 0.35794544,
                                    0.4101003, 0.44464612, 0.4826307, 0.4889217, 0.6693059, 0.6870472, 0.82553387, 0.86639106};

std::vector<float> key_x_golden = {0.5040901, 0.51998615, 0.48847437, 0.5424121, 0.4656546, 0.57788885, 0.43163484, 0.6175001, 0.38216457,
                                    0.60705376, 0.4151504, 0.554312, 0.45635343, 0.54465926, 0.45363468, 0.5266466, 0.42971918};

std::vector<float> score_golden = {0.44470087, 0.5480462, 0.6591966, 0.3904021, 0.4450525, 0.7143093, 0.6618434, 0.5292502, 0.689163, 
                                    0.48138157, 0.64457417, 0.71147513, 0.6135119, 0.51524675, 0.6072499, 0.5642306, 0.5547067};


class DetectTest : public tflite_shims::testing::Test {};

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

TEST_F(DetectTest, SucceedsWithFloatModel) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("womanstanding.jpg"));
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
  float y = result.landmarks(0).key_y();
  EXPECT_EQ(y, key_y_golden[0]);


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
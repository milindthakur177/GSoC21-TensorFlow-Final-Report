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

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";
constexpr char kMobileNetFloatWithMetadata[] =
    "lite-model_movenet_singlepose_lightning_tflite_int8_4_with_metadata.tflite";

std::vector<float> key_y_golden = {0.5010699, 0.52654934, 0.47475347, 0.5659141, 0.44451794, 0.6487602, 0.35149667, 0.6574936,
                        0.3209864, 0.54254323, 0.52659225, 0.5792549, 0.42052758, 0.62838054, 0.40062594, 0.49748933, 0.6251471};

std::vector<float> key_x_golden = {0.3613621, 0.33323765, 0.33484635, 0.3527827, 0.3565011, 0.4915269, 0.48380172, 0.74440265, 
                        0.7394606, 0.69045323, 0.69133437, 0.813216, 0.81319857, 0.8274471, 0.8424358,  0.7112423, 0.80640984};

std::vector<float> score_golden = {0.56745684, 0.7113907, 0.5633223, 0.59997165, 0.7448181, 0.81670046, 0.8441073, 0.85803306, 
                        0.84626555, 0.35415077, 0.5010598, 0.6837475, 0.69535846, 0.15943679, 0.07926878, 0.10836774, 0.07497841};

class DetectTest : public tflite_shims::testing::Test {};

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

TEST_F(DetectTest, SucceedsWithFloatModel) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("img.jpg"));
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
  float y = results.landmarks(0).key_y();
  EXPECT_EQ(y, key_y_golden[0]);

/*
  EXPECT_THAT(
      result,
      R"pb( landmarks {key_x : 0.3613621 key_y : 0.5010699 score : 0.56745684}
            landmarks {key_x : 0.33323765 key_y : 0.52654934 score : 0.7113907}
            landmarks {key_x : 0.33484635 key_y : 0.47475347 score : 0.5633223}
            landmarks {key_x : 0.3527827 key_y : 0.5659141 score : 0.59997165}
            landmarks {key_x : 0.3565011 key_y : 0.44451794 score : 0.7448181}
            landmarks {key_x : 0.4915269 key_y : 0.6487602 score : 0.81670046}
            landmarks {key_x : 0.48380172 key_y : 0.35149667 score : 0.8441073}
            landmarks {key_x : 0.74440265 key_y : 0.6574936 score : 0.85803306}
            landmarks {key_x : 0.7394606 key_y : 0.3209864 score : 0.84626555}
            landmarks {key_x : 0.69045323 key_y : 0.54254323 score : 0.35415077}
            landmarks {key_x : 0.69133437 key_y : 0.52659225 score : 0.5010598}
            landmarks {key_x : 0.813216 key_y : 0.5792549 score : 0.6837475}
            landmarks {key_x : 0.81319857 key_y : 0.42052758 score : 0.69535846}
            landmarks {key_x : 0.8274471 key_y : 0.62838054 score : 0.15943679}
            landmarks {key_x : 0.8424358 key_y : 0.40062594 score : 0.07926878}
            landmarks {key_x : 0.7112423 key_y : 0.49748933 score : 0.10836774}
            landmarks {key_x : 0.80640984 key_y : 0.6251471 score : 0.07497841}
          )pb");

*/        
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
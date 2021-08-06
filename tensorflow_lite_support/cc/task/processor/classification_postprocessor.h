/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_

#include <initializer_list>

#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/classification_head.h"
#include "tensorflow_lite_support/cc/task/core/label_map_item.h"
#include "tensorflow_lite_support/cc/task/core/score_calibration.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/processor/processor.h"
#include "tensorflow_lite_support/cc/task/processor/proto/class.proto.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classification_options.proto.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classifications.proto.h"

namespace tflite {
namespace task {
namespace processor {

// This Postprocessor expects one output tensor with:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    -  `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or
//       `[1 x 1 x 1 x N]`
//    - optional (but recommended) label map(s) as AssociatedFile-s with type
//      TENSOR_AXIS_LABELS, containing one label per line. The first such
//      AssociatedFile (if any) is used to fill the `class_name` field of the
//      results. The `display_name` field is filled from the AssociatedFile (if
//      any) whose locale matches the `display_names_locale` field of the
//      `ImageClassifierOptions` used at creation time ("en" by default, i.e.
//      English). If none of these are available, only the `index` field of the
//      results will be filled.
class ClassificationPostprocessor : public Postprocessor {
 public:
  static absl::StatusOr<std::unique_ptr<ClassificationPostprocessor>> Create(
      core::TfLiteEngine* engine,
      const std::initializer_list<int> output_indices,
      std::unique_ptr<ClassificationOptions> options) {
    RETURN_IF_ERROR(Postprocessor::SanityCheck(/* num_expected_tensors = */ 1,
                                               engine, output_indices));

    auto processor = absl::WrapUnique(
        new ClassificationPostprocessor(engine, output_indices));

    RETURN_IF_ERROR(processor->Init(std::move(options)));
    return processor;
  }

  template <typename T>
  absl::Status Postprocess(T* classifications);

 private:
  using Postprocessor::Postprocessor;

  absl::Status Init(std::unique_ptr<ClassificationOptions> options);
  // Given a ClassificationResult object containing class indices, fills the
  // name and display name from the label map(s).
  template <typename T>
  absl::Status FillResultsFromLabelMaps(T* classifications);

  // The list of classification heads associated with the corresponding output
  // tensors. Built from TFLite Model Metadata.
  ::tflite::task::core::ClassificationHead classification_head_{};

  // Set of allowlisted or denylisted class names.
  struct ClassNameSet {
    absl::flat_hash_set<std::string> values;
    bool is_allowlist;
  };

  // Allowlisted or denylisted class names based on provided options at
  // construction time. These are used to filter out results during
  // post-processing.
  ClassNameSet class_name_set_;

  // Score calibration parameters, if any. Built from TFLite Model
  // Metadata.
  std::unique_ptr<core::ScoreCalibration> score_calibration_;

  // Number of classes returned by `Postprocess` method.
  int num_results_;

  // Recommended score threshold typically in [0,1[. Classification results with
  // a score below this value are considered low-confidence and should be
  // rejected from returned results.
  float score_threshold_;

  // Default score value used as a fallback for classes that (1) have no score
  // calibration data or (2) have a very low confident uncalibrated score, i.e.
  // lower than the `min_uncalibrated_score` threshold.
  //
  // (1) This happens when the ScoreCalibration does not cover all the classes
  // listed in the label map. This can be used to enforce the denylisting of
  // given classes so that they are never returned.
  //
  // (2) This is an optional threshold provided part of the calibration data. It
  // is used to mitigate false alarms on some classes.
  //
  // In both cases, a class that gets assigned a score of -1 is never returned
  // as it gets discarded by the `score_threshold` check (see post-processing
  // logic).
  static constexpr float kDefaultCalibratedScore = -1.0f;

  // Calibrated scores should be in the [0, 1] range, otherwise an error is
  // returned at post-processing time.
  static constexpr float kMinCalibratedScore = 0.0f;
  static constexpr float kMaxCalibratedScore = 1.0f;
};

template <typename T>
absl::Status ClassificationPostprocessor::Postprocess(T* classifications) {
  classifications->set_head_index(output_indices_.at(0));
  std::vector<std::pair<int, float>> score_pairs;
  const auto& head = classification_head_;
  score_pairs.reserve(head.label_map_items.size());

  const TfLiteTensor* output_tensor = Tensor();
  if (output_tensor->type == kTfLiteUInt8) {
    const uint8* output_data =
        core::AssertAndReturnTypedTensor<uint8>(output_tensor);
    for (int j = 0; j < head.label_map_items.size(); ++j) {
      score_pairs.emplace_back(
          j, output_tensor->params.scale * (static_cast<int>(output_data[j]) -
                                            output_tensor->params.zero_point));
    }
  } else {
    const float* output_data =
        core::AssertAndReturnTypedTensor<float>(output_tensor);
    for (int j = 0; j < head.label_map_items.size(); ++j) {
      score_pairs.emplace_back(j, output_data[j]);
    }
  }

  // Optional score calibration.
  if (score_calibration_ != nullptr) {
    for (auto& score_pair : score_pairs) {
      const std::string& class_name =
          head.label_map_items[score_pair.first].name;
      score_pair.second = score_calibration_->ComputeCalibratedScore(
          class_name, score_pair.second);
      if (score_pair.second > kMaxCalibratedScore) {
        return support::CreateStatusWithPayload(
            absl::StatusCode::kInternal,
            absl::StrFormat("calibrated score is too high: got %f, expected "
                            "%f as maximum.",
                            score_pair.second, kMaxCalibratedScore));
      }
      if (score_pair.second != kDefaultCalibratedScore &&
          score_pair.second < kMinCalibratedScore) {
        return support::CreateStatusWithPayload(
            absl::StatusCode::kInternal,
            absl::StrFormat("calibrated score is too low: got %f, expected "
                            "%f as minimum.",
                            score_pair.second, kMinCalibratedScore));
      }
    }
  }

  if (class_name_set_.values.empty()) {
    // Partially sort in descending order (higher score is better).
    absl::c_partial_sort(
        score_pairs, score_pairs.begin() + num_results_,
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
          return a.second > b.second;
        });

    for (int j = 0; j < num_results_; ++j) {
      float score = score_pairs[j].second;
      if (score < score_threshold_) {
        break;
      }
      auto* cl = classifications->add_classes();
      cl->set_index(score_pairs[j].first);
      cl->set_score(score);
    }
  } else {
    // Sort in descending order (higher score is better).
    absl::c_sort(score_pairs, [](const std::pair<int, float>& a,
                                 const std::pair<int, float>& b) {
      return a.second > b.second;
    });

    for (int j = 0; j < head.label_map_items.size(); ++j) {
      float score = score_pairs[j].second;
      if (score < score_threshold_ ||
          classifications->classes_size() >= num_results_) {
        break;
      }

      const int class_index = score_pairs[j].first;
      const std::string& class_name = head.label_map_items[class_index].name;

      bool class_name_found = class_name_set_.values.contains(class_name);

      if ((!class_name_found && class_name_set_.is_allowlist) ||
          (class_name_found && !class_name_set_.is_allowlist)) {
        continue;
      }

      auto* cl = classifications->add_classes();
      cl->set_index(class_index);
      cl->set_score(score);
    }
  }
  return FillResultsFromLabelMaps(classifications);
}

template <typename T>
absl::Status ClassificationPostprocessor::FillResultsFromLabelMaps(
    T* classifications) {
  int head_index = classifications->head_index();
  const auto& label_map_items = classification_head_.label_map_items;
  for (int j = 0; j < classifications->classes_size(); ++j) {
    auto* current_class = classifications->mutable_classes(j);
    int current_class_index = current_class->index();
    if (current_class_index < 0 ||
        current_class_index >= label_map_items.size()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid class index (%d) with respect to label "
                          "map size (%d) for head #%d.",
                          current_class_index, label_map_items.size(),
                          head_index),
          support::TfLiteSupportStatus::kMetadataInconsistencyError);
    }
    const std::string& name = label_map_items[current_class_index].name;
    if (!name.empty()) {
      current_class->set_class_name(name);
    }
    const std::string& display_name =
        label_map_items[current_class_index].display_name;
    if (!display_name.empty()) {
      current_class->set_display_name(display_name);
    }
  }
  return absl::OkStatus();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_

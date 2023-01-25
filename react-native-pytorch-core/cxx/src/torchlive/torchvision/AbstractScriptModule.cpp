/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/jit/mobile/import.h>
#include <android/log.h>

#include "../torch/utils/helpers.h"
#include "ATen/core/ivalue.h"
#include "AbstractScriptModule.h"

namespace torchlive {
namespace torchvision {
namespace transforms {

using namespace facebook;

AbstractScriptModule::AbstractScriptModule(
    unsigned char* scriptedModule,
    unsigned int scriptModuleLength)
    : scriptmodule_(loadScriptModule(scriptedModule, scriptModuleLength)) {}

torch_::jit::mobile::Module AbstractScriptModule::loadScriptModule(
    unsigned char* scriptedModule,
    unsigned int scriptModuleLength) {
  std::stringstream is;
  is.write((char*)scriptedModule, scriptModuleLength);
  return torch_::jit::_load_for_mobile(is, torch_::kCPU);
}

bool compareBoundingBoxes(std::array<float, 5> boxes1, std::array<float, 5> boxes2) {
  return boxes1[4] - boxes2[4];
}

bool IOU(std::array<float, 5> a, std::array<float, 5> b) {
  auto areaA = (a[2] - a[0]) * (a[3] - a[1]);
  if (areaA <= 0.0) return 0.0;

  auto areaB = (b[2] - b[0]) * (b[3] - b[1]);
  if (areaB <= 0.0) return 0.0;

  auto intersectionMinX = std::max(a[0], b[0]);
  auto intersectionMinY = std::max(a[1], b[1]);
  auto intersectionMaxX = std::min(a[2], b[2]);
  auto intersectionMaxY = std::min(a[3], b[3]);
  auto intersectionArea =
    std::max(intersectionMaxY - intersectionMinY, 0.0f) * std::max(intersectionMaxX - intersectionMinX, 0.0f);
  return intersectionArea / (areaA + areaB - intersectionArea);
}

c10::IValue AbstractScriptModule::nms(c10::IValue output) {
  at::Tensor tensor = output.toTensor();
  __android_log_print(ANDROID_LOG_ERROR, "leruo", "{1} output[0]=%ld", tensor.size(0));
  int rows = tensor.size(0);
  // float results[rows][5];
  std::vector<std::array<float, 5>> results;
  // int resultsSize = 0;

  float predictionThreshold = 0.8;
  float iOUThreshold = 0.3;
  int nMSLimit = 15;
  // TODO: PARAMS
  float imgScaleX = 1080 / 640;
  float imgScaleY = 1920 / 640;
  int startX = 0;
  int startY = 0;
  // TODO: PARAMS
  // auto accessor = tensor.accessor<float, 0>();
  // CLEAN UP LOW SCORES
  for (int i = 0; i < rows; i++) {
    auto outputs = tensor.index({0, i}).data_ptr<float>();
    // auto outputs = accessor[i].data();
    // Only consider an object detected if it has a confidence score of over predictionThreshold
    float score = outputs[4];
    __android_log_print(
      ANDROID_LOG_ERROR, "leruo",
      std::format("{:6}", (score > predictionThreshold + std::numeric_limits<float>::epsilon()))
    );
    __android_log_print(
      ANDROID_LOG_ERROR, "leruo",
      "{2} index=%d 0=%d 1=%d 2=%d 3=%d 4=%d",
      i,
      outputs[0],
      outputs[1],
      outputs[2],
      outputs[3],
      outputs[4]
    );
    if (score > predictionThreshold + std::numeric_limits<float>::epsilon()) {
      // Calulate the bound of the detected object bounding box
      auto x = outputs[0];
      auto y = outputs[1];
      auto w = outputs[2];
      auto h = outputs[3];

      auto left = imgScaleX * (x - w / 2);
      auto top = imgScaleY * (y - h / 2);

      std::array<float, 5> bounds = {
        startX + left,
        startY + top,
        w * imgScaleX,
        h * imgScaleY,
        score,
      };
      results.push_back(bounds);
      __android_log_print(
        ANDROID_LOG_ERROR, "leruo",
        "{2.1} results=%d score=%d",
        results.size(), score
      );
      // results[i][0] = startX + left;
      // results[i][1] = startY + top;
      // results[i][2] = w * imgScaleX;
      // results[i][3] = h * imgScaleY;
      // results[i][4] = score;
      // auto boundsTensor = torch_::zeros(4);
      // boundsTensor.index_put_({0}, startX + left);
      // boundsTensor.index_put_({1}, startY + top);
      // boundsTensor.index_put_({2}, w * imgScaleX);
      // boundsTensor.index_put_({3}, h * imgScaleY);

      // results.index_put_({i}, boundsTensor);
      // resultsSize++;
    }

    // results.resize_(resultsSize);
  }

  // PERFORM NMS
  int resultsSize = results.size();
  std::sort(results.begin(), results.end(), compareBoundingBoxes);
  at::Tensor resultsTensor = torch_::zeros({resultsSize, 5});
  int totalResults = 0;

  bool active[resultsSize];
  for(int i = 0; i < resultsSize; i++){
    active[i] = true;
  }
  int numActive = resultsSize;

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  bool done = false;
  for (int i = 0; i < resultsSize && !done; i++) {
    if (active[i]) {
      auto boxA = results[i];
      auto boundsTensor = torch_::zeros(5);
      boundsTensor.index_put_({0}, results[i][0]);
      boundsTensor.index_put_({1}, results[i][1]);
      boundsTensor.index_put_({2}, results[i][2]);
      boundsTensor.index_put_({3}, results[i][3]);
      boundsTensor.index_put_({4}, results[i][4]);
      __android_log_print(
        ANDROID_LOG_ERROR, "leruo",
        "{3} x=%d y=%d w=%d h=%d",
        results[i][0],
        results[i][1],
        results[i][2],
        results[i][3]
      );
      resultsTensor.index_put_({i}, boundsTensor);
      totalResults++;
      if (totalResults >= nMSLimit) break;

      for (int j = i + 1; j < resultsSize; j++) {
        if (active[j]) {
          auto boxB = results[j];
          if (IOU(boxA, boxB) > iOUThreshold) {
            active[j] = false;
            numActive -= 1;
            if (numActive <= 0) {
              done = true;
              break;
            }
          }
        }
      }
    }
  }

  // resultsTensor.resize_(totalResults);

  // __android_log_print(ANDROID_LOG_ERROR, "leruo", "{4} size[0]=%ld dim=%ld final_results=%d above_thresh_results=%d", resultsTensor.size(0), resultsTensor.dim(), totalResults, resultsSize);
  // std::stringstream ss1;
  // std::stringstream ss2;
  // assigning the value of num_float to ss1
  // ss1 << resultsTensor.size(0);

  // assigning the value of num_float to ss2
  // ss2 << resultsTensor.dim();

  // initializing two string variables with the values of ss1 and ss2
  // and converting it to string format with str() function
  // std::string str1 = ss1.str();
  // std::string str2 = ss2.str();
  // __android_log_print(ANDROID_LOG_ERROR, "leruo size[0]=%s dim=%s", ss1, ss2);
  // __android_log_print(ANDROID_LOG_ERROR, "leruo tensor=%s", resultsTensor.toString());
  // __android_log_print(ANDROID_LOG_ERROR, "leruo dim=%f" );

  return resultsTensor;
}

c10::IValue AbstractScriptModule::forward(
    std::vector<torch_::jit::IValue> inputs) {
  __android_log_print(ANDROID_LOG_ERROR, "leruo ", "forwarding");
  return AbstractScriptModule::nms(this->scriptmodule_.forward(inputs));
}

std::vector<torch_::jit::IValue> AbstractScriptModule::parseInput(
    facebook::jsi::Runtime& runtime,
    const facebook::jsi::Value& thisValue,
    const facebook::jsi::Value* arguments,
    size_t count) {
  if (count != 1) {
    throw jsi::JSError(
        runtime,
        "Module expect 1 input but " + std::to_string(count) + " are given.");
  }
  auto tensorHostObject =
      torchlive::utils::helpers::parseTensor(runtime, &arguments[0]);
  auto tensor = tensorHostObject->tensor;
  std::vector<torch_::jit::IValue> inputs;
  inputs.push_back(tensor);
  return inputs;
}

} // namespace transforms
} // namespace torchvision
} // namespace torchlive

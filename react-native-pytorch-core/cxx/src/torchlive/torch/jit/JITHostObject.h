/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <jsi/jsi.h>

namespace torchlive {
namespace torch {
namespace jit {

class JSI_EXPORT JITHostObject : public facebook::jsi::HostObject {
  facebook::jsi::Function _loadForMobile_;

 public:
  explicit JITHostObject(facebook::jsi::Runtime& runtime);

  facebook::jsi::Value get(
      facebook::jsi::Runtime&,
      const facebook::jsi::PropNameID& name) override;
  std::vector<facebook::jsi::PropNameID> getPropertyNames(
      facebook::jsi::Runtime& rt) override;

 private:
  static facebook::jsi::Function create_LoadForMobile(
      facebook::jsi::Runtime& runtime);
};

} // namespace jit
} // namespace torch
} // namespace torchlive

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import SwiftyJSON

class TensorListUnpacker: Unpacker {
    func unpack(ivalue: IValue, modelSpec: JSON, result: inout [String: Any], packerContext: PackerContext) throws {
        let tensors = ivalue.toTensorList()
        for (index, tensor) in tensors!.enumerated() {
            let array = tensor.getDataAsArray() ?? []
            result[String(index)] = array
        }
    }
}


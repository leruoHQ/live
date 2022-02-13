/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import SwiftyJSON

class YoloV5Unpacker: Unpacker {
    func unpack(ivalue: IValue, modelSpec: JSON, result: inout [String: Any], packerContext: PackerContext) throws {
        let unpack = modelSpec["unpack"]

        
        guard let key = unpack["key"].string else {
            throw BaseIValuePackerError.missingKeyParam
        }
        guard let tensor = ivalue.toTensor() else {
            throw BaseIValuePackerError.decodeObjectsError
        }
        let probabilityThreshold = unpack["probabilityThreshold"].doubleValue
        let predictionsShape = tensor.shape
        
        guard let predictionsTensor = tensor.getDataAsArray() else {
            throw BaseIValuePackerError.decodeObjectsError
        }

        var boxedResults = [Any]()
        for index in 0...(predictionsShape[1].intValue) {
            let confidence = predictionsTensor[index * 6 + 4].doubleValue * predictionsTensor[index * 6 + 5].doubleValue; // conf = obj_conf * cls_conf
            if (confidence >= probabilityThreshold) {
                var match = [String: Any]()

                var bounds = [Double]()
                bounds.append(predictionsTensor[index * 6].doubleValue)
                bounds.append(predictionsTensor[index * 6 + 1].doubleValue)
                bounds.append(predictionsTensor[index * 6 + 2].doubleValue)
                bounds.append(predictionsTensor[index * 6 + 3].doubleValue)
                match["bounds"] = bounds
                match["confidence"] = predictionsTensor[index * 6 + 4]

                boxedResults.append(match)
            }
        }

        result[key] = boxedResults
    }
}


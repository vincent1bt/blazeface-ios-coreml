//
//  Utils.swift
//  faceRecognitionCoreML
//
//  Created by vicente rodriguez on 14/04/21.
//

import Foundation
import Vision

let boxesCount = 896

func getArrays(boxes: MLMultiArray) -> (xArray: [Float], yarray: [Float], width: [Float], height: [Float]) {
    var xArray: [Float] = []
    var yarray: [Float] = []
    var width: [Float] = []
    var height: [Float] = []

    for i in 0..<boxesCount {
        xArray.append(boxes[4 * i].floatValue)
        yarray.append(boxes[4 * i + 1].floatValue)
        width.append(boxes[4 * i + 2].floatValue)
        height.append(boxes[4 * i + 3].floatValue)
    }
    
    return (xArray, yarray, width, height)
}

//
//  ContentView.swift
//  FaceRecognitionAppTF
//
//  Created by vicente rodriguez on 19/01/21.
//

import SwiftUI
import AVFoundation
import UIKit
import Vision

struct ContentView: View {
    var body: some View {
        CameraView()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct FrameView: View {
    var boxes: [BoxPrediction] = []

    var body: some View {
        ForEach(boxes, id: \.id) { box in
            Rectangle().strokeBorder(Color.red).frame(width: box.rect.width, height: box.rect.height).position(x: box.rect.midX, y: box.rect.midY)
        }
    }
}

struct CameraView: View {
    @StateObject var camera = CameraModel()
    
    var body: some View {
        ZStack {
            CameraPreview(camera: camera).ignoresSafeArea(.all, edges: .all)
            FrameView(boxes: camera.boxes)
            Rectangle().strokeBorder(Color.blue).frame(width: camera.maxWidth, height: camera.maxHeight)
            Text(String(camera.frames)).position(x: 100, y: 100)
        }.onAppear(perform: {
            camera.Check()
        })
    }
}

class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var isTaken = false
    @Published var session = AVCaptureSession()
    
    @Published var alert = false
    @Published var output = AVCaptureVideoDataOutput()
    
    @Published var preview: AVCaptureVideoPreviewLayer!
    
    private let AMS = AverageMaximumSuppresion()
    
    @Published var boxes: [BoxPrediction] = []
    
    @Published var frames: Int = 60
    
    var maxWidth: CGFloat = 0
    var maxHeight: CGFloat = 0
    var cameraSizeRect: CGRect = CGRect.zero
    
    var imageWidth: Int = 1080
    var imageHeight: Int = 1920
    
    var classificationRequest: VNCoreMLRequest {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        
        do {
            let model = try Face500(configuration: config)
            let visionModel = try VNCoreMLModel(for: model.model)
            
            let visionRequest = VNCoreMLRequest(model: visionModel) { request, error in
                
                guard let results = (request.results as? [VNCoreMLFeatureValueObservation]) else {
                    fatalError("Unexpected result type from VNCoreMLRequest")
                }
                
                guard let predictions = results[0].featureValue.multiArrayValue else {
                    fatalError("Result 0 is not a MultiArray")
                }
                
                var arrayPredictions: [Float] = []
                
                for i in 0..<predictions.count {
                    arrayPredictions.append(predictions[i].floatValue)
                }
                
                guard let boxes = results[1].featureValue.multiArrayValue else {
                    fatalError("Result 1 is not a MultiArray")
                }
                
                let finalBoxes = self.getFinalBoxes(boxes: boxes, arrayPredictions: arrayPredictions)
                
                DispatchQueue.main.async {
                    self.boxes = finalBoxes
                }
    
            }
            
            visionRequest.imageCropAndScaleOption = .centerCrop
            return visionRequest
            
        } catch {
            fatalError("Failed to load ML model: \(error)")
        }
    }
    
    func getFinalBoxes(boxes: MLMultiArray, arrayPredictions: [Float]) -> [BoxPrediction] {
        
        let arrays = getArrays(boxes: boxes)
        
        let finalBoxes: [BoxPrediction] = AMS.getFinalBoxes(rawXArray: arrays.xArray, rawYArray: arrays.yarray, rawWidthArray: arrays.width, rawHeightArray: arrays.height, classPredictions: arrayPredictions, imageWidth: Float(imageWidth), imageHeight: Float(imageHeight), cameraSize: cameraSizeRect)
        
        return finalBoxes
    }
    
    func Check() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setUp()
            return
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { (status) in
                if status {
                    self.setUp()
                }
            }
        case .denied:
            self.alert.toggle()
            return
        default:
            return
        }
    }
    
    func setUp() {
        do {
            self.session.beginConfiguration()
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            
            let input = try AVCaptureDeviceInput(device: device!)
            
            if self.session.canAddInput(input) {
                self.session.addInput(input)
            }
            
            self.output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sample buffer"))
            self.output.alwaysDiscardsLateVideoFrames = true
            self.output.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey) : kCMPixelFormat_32BGRA]
            
            if self.session.canAddOutput(self.output) {
                self.session.addOutput(self.output)
                self.output.connection(with: .video)?.videoOrientation = .portrait
            }
            
            self.session.commitConfiguration()
            
        } catch {
            print(error.localizedDescription)
        }
        
        cameraSizeRect = preview.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        maxHeight = cameraSizeRect.width
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let pixelBuffer: CVPixelBuffer? = CMSampleBufferGetImageBuffer(sampleBuffer)
        
        guard let imagePixelBuffer = pixelBuffer else {
            return
        }
        
        imageWidth = CVPixelBufferGetWidth(imagePixelBuffer)
        imageHeight = CVPixelBufferGetHeight(imagePixelBuffer)
        
        runModel(bufferImage: imagePixelBuffer)
    }
    
    func runModel(bufferImage: CVPixelBuffer) {
        let startTime = CACurrentMediaTime()
        
        let handler = VNImageRequestHandler(cvPixelBuffer: bufferImage, orientation: .up)
        
        do {
            try handler.perform([classificationRequest])
        } catch {
            print("Failed to perform classification: \(error.localizedDescription)")
        }
        
        let finalTime = CACurrentMediaTime()
        
        let fullComputationFrames = 1 / (finalTime - startTime)
        
        DispatchQueue.main.async {
            self.frames = Int(fullComputationFrames)
        }
        
    }
}

struct CameraPreview: UIViewRepresentable {
    @ObservedObject var camera: CameraModel
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        camera.preview = AVCaptureVideoPreviewLayer(session: camera.session)
        camera.preview.frame = view.frame
        camera.preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(camera.preview)
        
        camera.session.startRunning()
        
        camera.maxWidth = camera.preview.bounds.size.width
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
    }
}

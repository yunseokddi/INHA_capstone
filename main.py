import os
import torch
import time

from classifier import Classifier
from inference import Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    '''
    ------------------------------------------
    |                Test                    |
    ------------------------------------------
    '''
    # harness = Classifier(phase='test',
    #                      data_path='./sample_data/test',
    #                      weight_path='./weight/experiment_1_256_weight.pth',
    #                      epoch=1, batch_size=4)
    #
    # best_acc = harness.test()

    '''
    ------------------------------------------
    |           Image Inference              |
    ------------------------------------------
    '''
    # detector = Detector(weight_path='./weight/experiment_1_256_weight.pth',
    #                     device=device, batch_size=16)
    #
    # # Inference for image batch
    # detector.detect(image_path='./inference_data/normal/', mode='batch')
    #
    # # Inference for single image
    # detector.detect(image_path='./inference_data/normal/sample.jpg', mode='single')

    '''
    ------------------------------------------
    |           Video Inference              |
    ------------------------------------------
    '''
    detector = Detector(weight_path='./weight/experiment_1_256_weight.pth',
                        device=device, batch_size=16)
    # Inference for video file
    detector.detect_video(video_path='./inference_data/sample.mp4', mode='video', FPS=8)

    # Inference for webcam
    # detector.detect_video(video_path='./inference_data/sample.mp4', mode='webcam', FPS=8)
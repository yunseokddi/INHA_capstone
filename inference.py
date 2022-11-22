import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os
import cv2
import time

from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from dataloader import InferDataset
from datetime import datetime


class Detector(object):
    def __init__(self, weight_path, device, batch_size):
        self.weight_path = weight_path
        self.device = device
        self.batch_size = batch_size

        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2).cuda()
        self.model = nn.DataParallel(model).to(device=self.device)
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.eval()

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect(self, image_path, mode):
        if mode == "batch":
            image_list = os.listdir(image_path)
            image_name_list = []
            dataset = np.empty((1080, 1920, 3))

            for image in image_list:
                image_name_list.append(os.path.join(image_path, image))
                open_image = np.array(Image.open(os.path.join(image_path, image)))  # (1080, 1920, 3)
                dataset = np.append(dataset, open_image, axis=0)

            dataset = dataset.reshape((-1, 1080, 1920, 3))
            dataset = np.delete(dataset, [0], axis=0)  # (32, 1080, 1920, 3)

            infer_data = InferDataset(dataset)

            dataloader = DataLoader(dataset=infer_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

            with torch.no_grad():
                for input in dataloader:
                    input = input.float().cuda()
                    _, output = torch.max(self.model(input), 1)

                    # print(output.item())  # get prediction

        else:
            open_image = Image.open(image_path)  # (1080, 1920, 3)

            input = self.transformer(open_image).unsqueeze(0).float().cuda()

            _, output = torch.max(self.model(input), 1)

            # print(output.item())  # get prediction

    def detect_video(self, video_path, mode, FPS):
        labels = {
            0: 'abnormal',
            1: 'normal'
        }
        if mode == 'webcam':
            video = cv2.VideoCapture(0)

        else:
            video = cv2.VideoCapture(video_path)

        prev_time = 0

        while True:
            start = time.time()
            ret, frame = video.read()  # read the camera frame

            if (not ret):
                break

            current_time = time.time() - prev_time
            if current_time > 1. / FPS:
                prev_time = time.time()
                now = datetime.now()
                input = self.transformer(frame).unsqueeze(0).float().cuda()

                ret, output = torch.max(self.model(input), 1)

                label = labels[int(output.item())]

                print("Predict Label : {}".format(label))
                print("Inference time : {}".format(time.time() - start))
                print("Video date : {}".format(now))

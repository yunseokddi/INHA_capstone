import torch
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from test import test_model


class Classifier(object):
    def __init__(self, phase='train', data_path='/home/yunseok/Desktop/data/haesungDS', weight_path='./weights',
                 epoch=10, batch_size=64, lr=0.001, step_size=10, gamma=0.1, val_ratio=0.2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.phase = phase

        # ---------------------- hyper parameter setting ----------------------------
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.weight_path = weight_path
        self.val_ratio = val_ratio

        self.test_data_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dataset = ImageFolder(self.data_path, transform=self.test_data_transformer)
        print("Num of all dataset: {}".format(len(self.dataset)))

        self.dataloaders = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=4)

        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2).cuda()
        self.model = nn.DataParallel(model).to(device=self.device)

        if phase == 'test':
            self.model.load_state_dict(torch.load(self.weight_path))

    def test(self):
        best_acc = test_model(model=self.model, device=self.device, dataloader=self.dataloaders, dataset=self.dataset)

        return best_acc

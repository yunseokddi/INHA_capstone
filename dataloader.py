import torchvision.transforms as transforms

from torch.utils.data import Dataset

class InferDataset(Dataset):
    def __init__(self, pil_imgs):
        super(InferDataset, self, ).__init__()

        self.pil_imgs = pil_imgs
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pil_imgs)

    def __getitem__(self, idx):
        img = self.pil_imgs[idx]

        return self.transformer(img)
from torchvision import transforms
from torch.utils.data import Dataset

class CustomImage(Dataset):
    def __init__(self):
        super(CustomImage, self).__init__()
        self.transform = transforms.Compose([
                                            transforms.Resize(224),
                                            # transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]
                                                # mean=[0.5, 0.5, 0.5],
                                                # std=[0.5, 0.5, 0.5]
                                            )])
    def __call__(self, Input_img):
        Input_img = Input_img.convert('RGB')
        return self.transform(Input_img)

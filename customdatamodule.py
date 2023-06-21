import lightning as L
from typing import Optional
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path="testing_jargon/face_split",
        batch_size=64,
        image_size=(218, 218),
        num_workers=0,
        logger_folder="lightning_logs",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.image_size = image_size
        self.logger_folder = logger_folder

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.level = stage
            trainset = ImageFolder(root=self.data_path+"/train", transform=self.transform)
            train_set_size = int(len(trainset) * 0.9)
            valid_set_size = len(trainset) - train_set_size
            self.train, self.validate = random_split(trainset, [train_set_size, valid_set_size])
        if stage == "test" or stage is None:
            self.level = stage
            self.test = trainset = ImageFolder(root=self.data_path+"/test", transform=self.transform)
    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.validate,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader

    @property
    def transform(self):
        if self.level == "fit":
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
        if self.level == "test":
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])

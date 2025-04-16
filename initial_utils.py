import csv
import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import glob

class SFRA_dataset(Dataset):
    def __init__(self, whether_is_training, resize):
        super(SFRA_dataset, self).__init__()
        self.whether_is_training = whether_is_training

        if self.whether_is_training:
            self.root = r".\training_dataset"
        else:
            self.root = r".\testing_dataset"

        self.whether_is_training = whether_is_training
        self.name2label = {}
        self.resize = resize
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            else:
                self.name2label[name] = len(self.name2label.keys())
        # gain, phase, label
        self.Gain_address_list, self.Phase_address_lists, self.label_lists = self.load_csv("images.csv")


    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, "*_Gain_*.png"))
                images += glob.glob(os.path.join(self.root, name, "*_Gain_*.jpg"))
                images += glob.glob(os.path.join(self.root, name, "*_Gain_*.jpeg"))
            # random
            random.shuffle(images)

            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    Gain = img
                    Phase = Gain.replace("Gain", "Phase")
                    writer.writerow([Gain, Phase, label])

        Gain_address_list, Phase_address_lists, label_lists = [], [], []
        with open(os.path.join(self.root, filename), mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                Gain, Phase, label = row
                label = int(label)
                Gain_address_list.append(Gain)
                Phase_address_lists.append(Phase)
                label_lists.append(label)
        assert len(Gain_address_list) == len(label_lists)
        if self.whether_is_training:
            print("Training set length:", len(Gain_address_list))
        else:
            print("Testing set length:", len(Gain_address_list))

        return Gain_address_list, Phase_address_lists, label_lists

    def __len__(self):
        return len(self.Gain_address_list)

    def __getitem__(self, item):
        Gain_address, Phase_address, label = self.Gain_address_list[item], self.Phase_address_lists[item], self.label_lists[item]

        tf = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        Gain, Phase = tf(Gain_address), tf(Phase_address)
        label = torch.tensor(label)
        return Gain, Phase, label



class selected_training_dataset(Dataset):
    def __init__(self, resize, training_dataset_address):
        super(selected_training_dataset, self).__init__()
        self.resize = resize
        self.training_dataset_address = training_dataset_address
        self.Gain_address_list, self.Phase_address_lists, self.label_lists = self.load_csv(self.training_dataset_address)

    def load_csv(self, filename):
        Gain_address_list, Phase_address_lists, label_lists = [], [], []
        with open(filename, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                Gain, Phase, label = row
                label = int(label)
                Gain_address_list.append(Gain)
                Phase_address_lists.append(Phase)
                label_lists.append(label)

        assert len(Gain_address_list) == len(label_lists)
        return Gain_address_list, Phase_address_lists, label_lists


    def __len__(self):
        return len(self.Gain_address_list)

    def __getitem__(self, item):
        Gain_address, Phase_address, label = self.Gain_address_list[item], self.Phase_address_lists[item], self.label_lists[item]

        tf = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        Gain, Phase = tf(Gain_address), tf(Phase_address)
        label = torch.tensor(label)
        return Gain, Phase, label


def test_autoencoder_dataset():
    write = SummaryWriter("logs")
    training_dataset = SFRA_dataset(whether_is_training=True, resize=128)
    a = DataLoader(training_dataset, batch_size=32, num_workers=0)
    for i, (Gain, Phase, label) in enumerate(a):
        write.add_images("Gain", Gain, i)
        write.add_images("Phase", Phase, i)
    write.close()


def test_selected_training_dataset():
    initial_dataset = selected_training_dataset(resize=128, training_dataset_address="./training_dataset_address_file/initial_training_dataset.csv")
    initial_dataloader = DataLoader(initial_dataset, batch_size=32, num_workers=0)
    for i, (Gain, Phase, label) in enumerate(initial_dataloader):
        print(i)

if __name__ == '__main__':
    test_selected_training_dataset()


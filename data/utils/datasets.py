import json
import os

os.environ["OMP_NUM_THREADS"] = '1'
import pickle
from argparse import Namespace
from pathlib import Path
from typing import List, Type, Dict
import wandb
import torch
import numpy as np
import torchvision
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.train_data_transform = None
        self.train_target_transform = None
        self.general_data_transform = None
        self.general_target_transform = None
        self.enable_train_transform = True

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class Synthetic(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CelebA(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).permute([0, -1, 1, 2]).float()
        self.targets = torch.from_numpy(targets).long()
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        self.classes = [0, 1]


class MedMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.classes = list(range(11))
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy")).float().unsqueeze(1)
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class COVID19(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy"))
            .permute([0, -1, 1, 2])
            .float()
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.classes = [0, 1, 2, 3]
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class USPS(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class MNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.MNIST(root, True, download=True)
        test_part = torchvision.datasets.MNIST(root, False)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class FashionMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.FashionMNIST(root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class EMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        split = None
        if isinstance(args, Namespace):
            split = args.emnist_split
        elif isinstance(args, dict):
            split = args["emnist_split"]
        train_part = torchvision.datasets.EMNIST(
            root, split=split, train=True, download=True
        )
        test_part = torchvision.datasets.EMNIST(
            root, split=split, train=False, download=True
        )
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        super_class = None
        if isinstance(args, Namespace):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in self.targets:
                new_targets.append(mapping[self.classes[cls]])
            self.targets = torch.tensor(new_targets, dtype=torch.long)


class TinyImagenet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = pd.read_table(
            root / "raw/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()

        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            mapping = dict(zip(self.classes, list(range(len(self.classes)))))
            data = []
            targets = []
            for cls in os.listdir(root / "raw" / "train"):
                for img_name in os.listdir(root / "raw" / "train" / cls / "images"):
                    img = pil_to_tensor(
                        Image.open(root / "raw" / "train" / cls / "images" / img_name)
                    ).float()
                    if img.shape[0] == 1:
                        img = torch.expand_copy(img, [3, 64, 64])
                    data.append(img)
                    targets.append(mapping[cls])

            table = pd.read_table(
                root / "raw/val/val_annotations.txt",
                sep="\t",
                engine="python",
                header=None,
            )
            test_classes = dict(zip(table[0].tolist(), table[1].tolist()))
            for img_name in os.listdir(root / "raw" / "val" / "images"):
                img = pil_to_tensor(
                    Image.open(root / "raw" / "val" / "images" / img_name)
                ).float()
                if img.shape[0] == 1:
                    img = torch.expand_copy(img, [3, 64, 64])
                data.append(img)
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CINIC10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            data = []
            targets = []
            mapping = dict(zip(self.classes, range(10)))
            for folder in ["test", "train", "valid"]:
                for cls in os.listdir(Path(root) / "raw" / folder):
                    for img_name in os.listdir(root / "raw" / folder / cls):
                        img = pil_to_tensor(
                            Image.open(root / "raw" / folder / cls / img_name)
                        ).float()
                        if img.shape[0] == 1:
                            img = torch.expand_copy(img, [3, 32, 32])
                        data.append(img)
                        targets.append(mapping[cls])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class DomainNet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/domain.sh` to download the dataset first."
            )
        targets_path = root / "targets.pt"
        metadata_path = root / "metadata.json"
        filename_list_path = root / "filename_list.pkl"
        if not (
            os.path.isfile(targets_path)
            and os.path.isfile(metadata_path)
            and os.path.isfile(filename_list_path)
        ):
            raise RuntimeError(
                "Run data/domain/preprocess.py to preprocess DomainNet first."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(filename_list_path, "rb") as f:
            self.filename_list = pickle.load(f)

        self.classes = list(metadata["classes"].keys())
        self.targets = torch.load(targets_path)
        self.pre_transform = transforms.Compose(
            [
                transforms.Resize([metadata["image_size"], metadata["image_size"]]),
                transforms.ToTensor(),
            ]
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform

    def __getitem__(self, index):
        data = self.pre_transform(Image.open(self.filename_list[index]).convert("RGB"))
        targets = self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets

import torch.utils.data as data
# class Dataset_Con_all_feedback_XD(data.Dataset):
#     def __init__(self,transform=None, test_mode=False):



#         if test_mode:

#             self.con_all = np.load("C:/Users/User/PycharmProjects/FL_AD/Concat_test_10.npy")
#             print('self.con_all shape:',self.con_all.shape)
#         else:
            
            
#             # self.con_all = np.load('iterative_UCF_labels/'+'{}.npy'.format(args.conall))
#             self.con_all = np.load('C:/Users/User/PycharmProjects/FL_AD/concat_UCF.npy')
#             print('self.con_all shape:',self.con_all.shape)
#             self.label_ab = np.load('C:/Users/User/PycharmProjects/FL_AD/Pseudo_Lebels_Maximum_AUC_0.8464948944321682.npy')
#             self.label_all = np.concatenate((np.zeros((779951-len(self.label_ab),)),self.label_ab), axis=0)
#             print('self.label_all shape:',self.label_all.shape)

#         self.tranform = transform
#         self.test_mode = test_mode
        

#     def __getitem__(self, index):


#         if self.test_mode:
#             features = self.con_all[index]
#             features = np.array(features, dtype=np.float32)
#             return features
#         else:
            
#             features = self.con_all[index]
#             features = np.array(features, dtype=np.float32)
#             labels = np.array(self.label_all[index], dtype=np.float32)

#             return features , labels
#     def __len__(self):
#         return len(self.con_all)

# functions

def estimate_gauss(X):
    m = X.shape[0]   # using only first dimension as we know it has only one feature - l2 norm
    
    mu = np.mean(X, axis=0)
    var = np.cov(X.T)
    
    return mu, var

def covariance_mat(X):
    X = np.mean(X , axis= 1)
    X =  X.transpose(1,0)
    cov  = np.cov(X)

    return cov

def get_matrix(data):

    l2_norm = np.sum(np.square(data), axis=2)
    n_train_crop_l2_norm_mean = np.mean(l2_norm, axis= 1)

    return n_train_crop_l2_norm_mean


def diff_l2(new_repr):

    l2_norms = []
    for i in range(len(new_repr)):
        l2_norms.append(get_matrix(new_repr[i]))

    mean_v_l2 = []
    for i in range(len(l2_norms)):
        mean_v_l2.append(np.diff(l2_norms[i], n=1).max())
    return mean_v_l2




def C2FPL_client(train_data,args,  client_partition,client_video_num_partition, client_id, total_clients, df, load):
    new_repr = []
    n_num = 0
    a_num = 0
    # for i  in range(len(client_partition['train'])):
    #     if client_video_num_partition['train'][i] > 809: 
    #         n_num += 1
    #     else:
    #         a_num += 1
    #     new_repr.append(train_data[client_partition['train'][i]])

    for i  in range(len(client_partition['train'])):
        new_repr.append(train_data[client_partition['train'][i]])
    if load == 0:

            # break
        # print(f"Client {client_id} has {n_num} normal videon and {a_num} abnormal videos \n Total number of videos of {n_num + a_num}")

        params = []
        top_k = 30
        for i in range(len(new_repr)):
            entropy = 0

            param_1 = get_matrix(new_repr[i]) # l2 norm
            mu, var = estimate_gauss(param_1) # mean and variance of l2 norm

            l2_diff = np.diff(param_1, n=1) # max diff
            var_diff = np.var(l2_diff) # variance of max diff
            max_diff = np.max(np.diff(param_1, n=1))
            param_2 = covariance_mat(new_repr[i])
            param_2 = np.where(param_2 == 0, 0.000000001,param_2)
            for i in np.diagonal(param_2):
                entropy += -(i * np.log(i))


            params.append((max_diff,  entropy))


        # params = []
        # top_k = 15
        # for i in range(len(new_repr)):
        #     entropy = 0

        #     param_1 = get_matrix(new_repr[i]) # l2 norm
        #     mu, var = estimate_gauss(param_1) # mean and variance of l2 norm

        #     l2_diff = np.diff(param_1, n=1) # max diff
        #     var_diff = np.var(l2_diff) # variance of max diff

        #     param_2 = covariance_mat(new_repr[i]) # covariance matrix
        #     cov_var_sum = np.sum(np.diag(param_2))  # sum of diagonal elements
        #     param_2 = np.where(param_2 == 0, 0.000000001,param_2)
        #     # param_2 = param_2[~np.isnan(param_2)]
        #     s= np.linalg.eigvals(param_2) # singular values
        #     for i in s[:top_k]:
        #         entropy += (i * np.log(i))  # entropy of covariance matrix


        #     params.append((var_diff, entropy))

        # params = []
        # top_k = 30
        # for i in range(len(new_repr) ):
        #     param_1 = get_matrix(new_repr[i])
        #     mu, var = estimate_gauss(param_1)
        #     param_2 = covariance_mat(new_repr[i])

        #     params.append(np.diagonal(param_2[:top_k, :top_k]))


        # params = []

        # for i in range(len(new_repr)):

        #     param = get_matrix(new_repr[i])
        #     mu, var = estimate_gauss(param)


        #     params.append((mu,var,))




        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # sample = pca.fit(params).transform(params)




        from sklearn.mixture import GaussianMixture
    


        gmm = GaussianMixture(n_components=2, max_iter=150, random_state=0)

        labels = gmm.fit_predict(params)

        import matplotlib.pyplot as plt

        # plt.scatter(sample[:, 0], sample[:, 1], c= labels)
        # plt.title(f"Client {client_id} Clusters")
        # wandb.log({f"Client {client_id} Clusters":  wandb.Image(plt)})


        import pandas as pd


        sum_normal = 0
        sum_abnormal = 0
        c_normal = 0
        c_abnormal = 0

        set_1 = {}
        set_2 = {}
        for i in range(len(new_repr)):
            if labels[i] == 0:
                set_1[client_video_num_partition['train'][i]] = new_repr[i]
                sum_abnormal += params[i][1]
                c_abnormal += 1
            else:
                set_2[client_video_num_partition['train'][i]] = new_repr[i]
                sum_normal += params[i][1]
                c_normal += 1





        if len(set_1.keys()) > len(set_2.keys()):
            normal_set = set_1
            abnormal_set = set_2  
        else:
            normal_set = set_2
            abnormal_set = set_1        


        normal_l2 = {}
        abnormal_l2 = {}
        for (idel, sample) in normal_set.items():
            normal_l2[idel] = get_matrix(sample)

        for (idel, sample) in abnormal_set.items():
            abnormal_l2[idel] = get_matrix(sample)

        abnormal_set_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_abnormal.pkl"
        with open(abnormal_set_path, "wb") as f: 
            pickle.dump(abnormal_set, f)
        abnormal_l2_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_abnormal_l2.pkl"
        with open(abnormal_l2_path, "wb") as f:
            pickle.dump(abnormal_l2, f)
        normal_l2_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_normal_l2.pkl"
        with open(normal_l2_path, "wb") as f:
            pickle.dump(normal_l2, f)

        l2_norms_N = np.empty(0,)
        client_sample_length = 0
        for (idel, sample) in normal_set.items():
            
            client_sample_length += len(sample)
            

            l2_norms_N = np.append(l2_norms_N,get_matrix(sample))



    else:
        print("Loading Clusters")
        #   all_abnormal  = {}

        cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_abnormal.pkl"
        with open(cluster_path, "rb") as f:
                abnormal_set = pickle.load(f)    
            # all_abnormal.update(abnormal_list)

        # all_normal  = {}
        

        cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_normal_l2.pkl"
        with open(cluster_path, "rb") as f:
                normal_set = pickle.load(f)   
        # all_normal.update(normal_list)  
        l2_norms_N = np.empty(0,)
        client_sample_length = 0
        for (idel, sample) in normal_set.items():
            
            client_sample_length += len(sample)
            

            l2_norms_N = np.append(l2_norms_N,sample)



    # n = 0
    # a = 0
    # for i in range(len(new_repr)):
        
    #     if client_video_num_partition['train'][i] in normal_set.keys():
    #         if client_video_num_partition['train'][i] > 809:
    #             # plt.scatter(params[i][0], params[i][1], c= "blue", marker="o")
    #             n += 1

    #         # else:
    #         #     # plt.scatter(params[i][0], params[i][1], c= "red", marker="o") 
    #     else:
    #         if client_video_num_partition['train'][i] <= 809:
    #             # plt.scatter(params[i][0], params[i][1], c= "blue", marker="x")
    #             a += 1
            # else:
            #     # plt.scatter(params[i][0], params[i][1], c= "red", marker="x") 
            
    # plt.title(f"Client {client_id} Clusters")
    # wandb.log({f"Client {client_id} Clusters":  wandb.Image(plt)})




    # df.loc[client_id, 'num_of_GT_normal'] = n_num
    # df.loc[client_id , 'num_of_GT_abnormal'] = a_num
    # df.loc[client_id, 'num_of_P_normal'] = len(normal_set.keys()) 
    # df.loc[client_id , 'num_of_P_abnormal'] = len(abnormal_set.keys())  
    # df.loc[client_id , 'Total_vids'] = n_num  + a_num
    # df.loc[client_id, 'correct_normal'] = n
    # df.loc[client_id , 'correct_abnormal'] = a 
    # df.loc[client_id, 'normal_acc %'] = n/len(normal_set.keys()) 
    # df.loc[client_id, 'abnormal_acc %'] = a/len(abnormal_set.keys()) 

    # print(df)

    wandb.log({f"Client {client_id} Clustering ACC": wandb.Table(dataframe=df)})



    mu_GMM, var_GMM = estimate_gauss(np.array(l2_norms_N))
    # probability model
    from scipy.stats import multivariate_normal
    p = multivariate_normal(mu_GMM, var_GMM)


    # ground_truth = {} 
    # length = 0.2 
    # for (idel, sample) in abnormal_set.items(): 

    #     # feature extraction 
    #     # sample_matrix = np.sum(np.square(sample), axis=1)  # for just l2
    #     sample_matrix = get_matrix(sample)
        
    #     # get p values
    #     probs = p.pdf(sample_matrix)
    #     temp_list = []
    #     temp_list += [0.0] * len(probs)
        
    #     window_size = int(len(probs) * length)  # fixed
    #     temp = []
    #     for idx in range(0, len(probs) - window_size + 1):
    #         arr = 0
    #         for i in range(idx, idx + window_size - 1):
    #             arr += abs(probs[i+1] - probs[i])
    #         temp.append(arr)
            

    #     for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
    #         temp_list[i] = 1.0

    #     ground_truth[idel] = temp_list



    # final_gt = {}
    # p = 0
    # for i in range(len(new_repr)):
    #     idx_in_all = client_video_num_partition['train'][i]
    #     if idx_in_all in normal_set.keys():
    #         p += 1
    #         final_gt[idx_in_all] = [0.0] * new_repr[i].shape[0]
    #     else:
    #         final_gt[idx_in_all] = ground_truth[idx_in_all]
    # print(p)
    return None, df, (mu_GMM, var_GMM, client_sample_length, list(normal_set.keys()))






def C2FPL_client_eta(train_data,args, client_partition,client_video_num_partition, client_id, total_clients, df, load):
    new_repr = []
    n_num = 0
    a_num = 0
    for i  in range(len(client_partition['train'])):
        if client_video_num_partition['train'][i] > 809: 
            n_num += 1
        else:
            a_num += 1
        new_repr.append(train_data[client_partition['train'][i]])
        # break
    # print(f"Client {client_id} has {n_num} normal videon and {a_num} abnormal videos \n Total number of videos of {n_num + a_num}")

    if load == 0:


        # params = []
        # top_k = 30
        # for i in range(len(new_repr)):
        #     entropy = 0

        #     param_1 = get_matrix(new_repr[i])
        #     mu, var = estimate_gauss(param_1)
        #     max_diff = np.max(np.diff(param_1, n=1))
        #     param_2 = covariance_mat(new_repr[i])
        #     param_2 = np.where(param_2 == 0, 0.000000001,param_2)
        #     for i in np.diagonal(param_2):
        #         entropy += -(i * np.log(i))


        #     params.append((max_diff,  entropy))




        params = []

        for i in range(len(new_repr)):

            param = get_matrix(new_repr[i])
            mu, var = estimate_gauss(param)


            params.append((mu,var,))




        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # sample = pca.fit(params).transform(params)




        from sklearn.mixture import GaussianMixture
    




        gmm = GaussianMixture(n_components=2, max_iter=150, random_state=0)
        # gmm_scores = gmm.score_samples(params)
        # labels = gmm.fit_predict(params)

        # y_gmm = gmm.fit_predict(params)
        # # print(y_gmm.sum(), y_gmm.sum() / len(y_gmm))

        # score = y_gmm 
        # score = gmm.score_samples(params) 
        # pct_threshold = np.percentile(score, 4)
        # print(f'The threshold of the score is {pct_threshold:.2f}') 
        # res = np.array([1 if x < pct_threshold else 0 for x in score]) 
        res = gmm.fit_predict(params)
        # print(res.sum())


        abnormal_portion = np.where(res == 1)[0]
        normal_portion = np.where(res == 0)[0]
        normal_portion.shape, abnormal_portion.shape

        abag = list(zip(list(np.array(params)[abnormal_portion]), abnormal_portion))
        nbag = list(zip(list(np.array(params)[normal_portion]), normal_portion))



        import warnings
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

        nu = 1.0
        step = 1


        while len(abag) / len(nbag) < nu:
            
            temp_bag = nbag
            res = gmm.fit_predict([list(x[0]) for x in np.array(temp_bag)])

            abnormal_portion = np.where(res == 1)[0]
            normal_portion = np.where(res == 0)[0]
            
            abag += [(x[0], x[1]) for x in np.array(temp_bag)[abnormal_portion]]
            nbag = [(x[0], x[1]) for x in np.array(temp_bag)[normal_portion]]

            step += 1


        print("abag length", len(abag), "nbag length", len(nbag))
        # nu = 1.0
        # step = 1
        # import time
        # # start = time.time()
        # while len(abag) / len(nbag) < nu:
            
        #     temp_bag = nbag
        #     y_gmm = gmm.fit_predict([list(x[0]) for x in np.array(temp_bag)])
        #     score = y_gmm 
        #     score = gmm.score_samples([list(x[0]) for x in np.array(temp_bag)]) 
        #     pct_threshold = np.percentile(score, 4) 
        #     res = np.array([1 if x < pct_threshold else 0 for x in score]) 
        #     # print(f'The threshold of the score in step {step} is {pct_threshold:.2f}, abnormal part: {res.sum()}') 
            
        #     abnormal_portion = np.where(res == 1)[0]
        #     normal_portion = np.where(res == 0)[0]
            
        #     abag += [(x[0], x[1]) for x in np.array(temp_bag)[abnormal_portion]]
        #     nbag = [(x[0], x[1]) for x in np.array(temp_bag)[normal_portion]]

        #     step += 1

        # print('correctness acc: ', np.where(np.array([client_video_num_partition['train'][x[1]] for x in abag]) < 809)[0].shape[0] / len([x[1] for x in abag]))


        # print('correctness acc: ', np.where(np.array([client_video_num_partition['train'][x[1]] for x in nbag]) > 809)[0].shape[0] / len([x[1] for x in nbag]))

        # temp = [k[1] for k in sorted([(x[1], 1.0) for x in abag] + [(x[1], 0.0) for x in nbag], key=lambda z: z[0])]


        import matplotlib.pyplot as plt
        import pandas as pd
        sum_normal = 0
        sum_abnormal = 0
        c_normal = 0
        c_abnormal = 0
        set_a = {}
        set_n = {}
        normal_set = {}
        abnormal_set = {}
        for i in abag:
            set_a[client_video_num_partition['train'][i[1]]] = new_repr[i[1]]
            sum_abnormal += params[i[1]][1]
            c_abnormal += 1
            # cluster_df.loc[len(df)] = [client_video_num_partition['train'][i[1]], "Abnormal", max(params[i[1]]), np.mean(params[i[1]])] 

        for i in nbag:
            set_n[client_video_num_partition['train'][i[1]]] = new_repr[i[1]]
            sum_normal += params[i[1]][1]
            c_normal += 1
            # cluster_df.loc[len(df)] = [client_video_num_partition['train'][i[1]], "Normal", max(params[i[1]]), np.mean(params[i[1]])] 

        # if np.real(sum_normal / c_normal) > np.real(sum_abnormal / c_abnormal):
        #     normal_set = set_n
        #     abnormal_set = set_a
        # else:
        #     normal_set = set_a
        #     abnormal_set = set_n
 

        




        normal_l2 = {}
        abnormal_l2 = {}
        for (idel, sample) in normal_set.items():
            normal_l2[idel] = get_matrix(sample)

        for (idel, sample) in abnormal_set.items():
            abnormal_l2[idel] = get_matrix(sample)

        with open(f"data/ucf/clusters/eta_{client_id}_of_{total_clients}_abnormal.pkl", "wb") as f:
            pickle.dump(abnormal_set, f)

        with open(f"data/ucf/clusters/eta_{client_id}_of_{total_clients}_abnormal_l2.pkl", "wb") as f:
            pickle.dump(abnormal_l2, f)

        with open(f"data/ucf/clusters/eta_{client_id}_of_{total_clients}_normal_l2.pkl", "wb") as f:
            pickle.dump(normal_l2, f)
        

        l2_norms_N = np.empty(0,)
        client_sample_length = 0
        for (idel, sample) in normal_set.items():
            
            client_sample_length += len(sample)
            

            l2_norms_N = np.append(l2_norms_N,get_matrix(sample))



        # for i in range(len(new_repr)):
        #     if temp[i] == 0.0:
        #         normal_set[client_video_num_partition['train'][i]] = new_repr[i]
        #     else:
        #         abnormal_set[client_video_num_partition['train'][i]] = new_repr[i]

        # if len(set_1.keys()) > len(set_2.keys()):
        #     normal_set = set_1
        #     abnormal_set = set_2  
        # else:
        #     normal_set = set_2
        #     abnormal_set = set_1        

        # with open(f"C:/Users/User/PycharmProjects/FL_AD/data/ucf/clusters/{client_id}_abnormal.pkl", "wb") as f:
        #     pickle.dump(abnormal_set, f)

        # with open(f"C:/Users/User/PycharmProjects/FL_AD/data/ucf/clusters/{client_id}_normal.pkl", "wb") as f:
        #     pickle.dump(normal_set, f)


        n = 0
        a = 0
        for i in range(len(new_repr)):
            
            if client_video_num_partition['train'][i] in normal_set.keys():
                if client_video_num_partition['train'][i] > 809:
                    # plt.scatter(params[i][0], params[i][1], c= "blue", marker="o")
                    n += 1

                # else:
                #     # plt.scatter(params[i][0], params[i][1], c= "red", marker="o") 
            else:
                if client_video_num_partition['train'][i] <= 809:
                    # plt.scatter(params[i][0], params[i][1], c= "blue", marker="x")
                    a += 1
                # else:
                #     # plt.scatter(params[i][0], params[i][1], c= "red", marker="x") 
                
        # plt.title(f"Client {client_id} Clusters")
        # wandb.log({f"Client {client_id} Clusters":  wandb.Image(plt)})




        # df.loc[client_id, 'num_of_GT_normal'] = n_num
        # df.loc[client_id , 'num_of_GT_abnormal'] = a_num
        # df.loc[client_id, 'num_of_P_normal'] = len(normal_set.keys()) 
        # df.loc[client_id , 'num_of_P_abnormal'] = len(abnormal_set.keys())  
        # df.loc[client_id , 'Total_vids'] = n_num  + a_num
        # df.loc[client_id, 'correct_normal'] = n
        # df.loc[client_id , 'correct_abnormal'] = a 
        # df.loc[client_id, 'normal_acc %'] = n/len(normal_set.keys()) 
        # df.loc[client_id, 'abnormal_acc %'] = a/len(abnormal_set.keys()) 

        # print(df)

        # wandb.log({f"Client {client_id} Clustering ACC": wandb.Table(dataframe=df)})

    else:
        print("Loading Clusters")
        #   all_abnormal  = {}

        cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"eta_{client_id}_of_{total_clients}_abnormal.pkl"
        with open(cluster_path, "rb") as f:
                abnormal_set = pickle.load(f)    
            # all_abnormal.update(abnormal_list)

        # all_normal  = {}
        

        cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"eta_{client_id}_of_{total_clients}_normal_l2.pkl"
        with open(cluster_path, "rb") as f:
                normal_set = pickle.load(f)   
        # all_normal.update(normal_list)  
        l2_norms_N = np.empty(0,)
        client_sample_length = 0
        for (idel, sample) in normal_set.items():
            
            client_sample_length += len(sample)
            

            l2_norms_N = np.append(l2_norms_N,sample)

    # l2_norms_N = np.empty(0,)
    # for (idel, sample) in normal_set.items():
        
    #     # print(sample.shape)
        

    #     l2_norms_N = np.append(l2_norms_N,get_matrix(sample))

    mu_GMM, var_GMM = estimate_gauss(np.array(l2_norms_N))
    # probability model
    from scipy.stats import multivariate_normal
    p = multivariate_normal(mu_GMM, var_GMM)


    ground_truth = {} 
    length = 0.2 
    for (idel, sample) in abnormal_set.items(): 

        # feature extraction 
        # sample_matrix = np.sum(np.square(sample), axis=1)  # for just l2
        sample_matrix = get_matrix(sample)
        
        # get p values
        probs = p.pdf(sample_matrix)
        temp_list = []
        temp_list += [0.0] * len(probs)
        
        window_size = int(len(probs) * length)  # fixed
        temp = []
        for idx in range(0, len(probs) - window_size + 1):
            arr = 0
            for i in range(idx, idx + window_size - 1):
                arr += abs(probs[i+1] - probs[i])
            temp.append(arr)
            

        for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
            temp_list[i] = 1.0

        ground_truth[idel] = temp_list



    final_gt = {}

    for i in normal_set.keys():
        final_gt[i] = [0.0] * len(normal_set[i])

    for i in abnormal_set.keys():
        final_gt[i] = ground_truth[i]    


    return final_gt, df, (mu_GMM, var_GMM, client_sample_length, list(normal_set.keys()))













PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()







def pl_refining(args, confidance_scores, total_clients):
    df_1 = pd.DataFrame(columns=['Class','max_confidance_score', 'mean_confidance_scores'])
    try:
        partition_path1 = PROJECT_DIR / "data" / args.dataset / args.partition
        partition_path2 = PROJECT_DIR / "data" / args.dataset / args.video_num_partition
        # cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_abnormal.pkl"
        with open(partition_path1, "rb") as f:
            client_partition = pickle.load(f)
        with open(partition_path2, "rb") as f:
            client_video_num_partition = pickle.load(f)
    except:
        raise FileNotFoundError(f"Please partition {args.dataset} first.")

    all_abnormal  = {}
    for client_id in range(total_clients):
        cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_abnormal_l2.pkl"
        with open(cluster_path, "rb") as f:
                abnormal_list = pickle.load(f)    
        all_abnormal.update(abnormal_list)

    original_pl = np.load("original_pl.npy")
    # with open(cluster_path, "rb") as f:
    #         abnormal_list = pickle.load(f)

    train_list = list(open('list/UCF_Train_ten_crop_i3d_complete_V1.txt'))
    fc_gt = []  # new pseudo-labels taken from fc network
    iou_scores = []     
    # for idel in range(len(train_list)):
    #     from_id = int(train_list[idel].split('\n')[0].split(',')[1])
    #     to_id = int(train_list[idel].split('\n')[0].split(',')[2])
    #     num_features = to_id - from_id
    for (idel, sample) in all_abnormal.items():
        class_type = 'Abnormal' if idel <= 809 else 'Normal' 
        

        num_features = len(sample)
        temp_list = []
        temp_list += [0.0] * num_features

        from_id = int(train_list[idel].split('\n')[0].split(',')[1])
        to_id = int(train_list[idel].split('\n')[0].split(',')[2])

        
        probs = confidance_scores[from_id:to_id]


        # df_1.loc[idel] = [class_type, np.max(confidance_scores[from_id:to_id]), np.mean(confidance_scores[from_id:to_id])]
        df_1.loc[idel, 'Class'] = class_type
        df_1.loc[idel, 'max_confidance_score'] = np.max(confidance_scores[from_id:to_id])
        df_1.loc[idel, 'mean_confidance_scores'] = np.mean(confidance_scores[from_id:to_id])
        
        wandb.log({f"Videos Confidance Scores": wandb.Table(dataframe=df_1)})

        
  


        window_size = int(num_features * 0.1)  # fixed
        temp = []
        for idx in range(0, num_features - window_size + 1):
            arr = 0
            for i in range(idx, idx + window_size - 1):
                arr += abs(probs[i+1] - probs[i])
            temp.append(arr)
        
        for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
            temp_list[i] = 1.0

        prev_lls = original_pl[from_id:to_id]
        next_lls = np.array(temp_list)
        res = np.where(prev_lls > 0.5, 1.0, 0.0) + next_lls
        if np.count_nonzero(res) > 0:
            iou = res[res == 2.].shape[0] / (res[res == 1.].shape[0] + res[res == 2.].shape[0])
            iou_scores.append(iou)
            with open('iou_scores.txt', 'a') as f:
                f.write(f"{iou}\n")
        else:
            print('no anomaly')

        new_rep = prev_lls + next_lls
        
        if iou < 0.05:
            new_rep[new_rep > 0.0] = 1.0 #np.max(probs) #
        else:
            new_rep /= 2    

        original_pl[from_id:to_id] = new_rep

        # fig, axs = plt.subplots(2, 1) 
        # axs[0].plot(probs)
        # axs[0].set_title(f"Video {idel} Confidance Scores and Pseudo-Labels")
        # axs[1].plot(new_rep)
        # # axs[1].set_title(f"Video {idel} Pseudo-Labels", loc='right')
        # wandb.log({f"Video {idel} Confidance Scores and Pseudo-Labels ":  wandb.Image(plt)})
        # plt.close()

    # save df_1 to csv
    df_1.to_csv('./confidance_scores_stats.csv', index=False)
    return np.array(original_pl), sum(iou_scores) / len(iou_scores)    



from itertools import islice
# dict(islice(abnormal_list.items(), 1, int(0.8*len(abnormal_list))))
def gmm_PL(args, total_clients, gmm_params, vids_num):

    all_abnormal  = {}
    for client_id in range(total_clients):
        if args.eta_clustering:
            cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"eta_{client_id}_of_{total_clients}_abnormal_l2.pkl"
        else:
            cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_abnormal_l2.pkl"
        with open(cluster_path, "rb") as f:
                abnormal_list = pickle.load(f)    
        all_abnormal.update(abnormal_list)

    all_normal  = {}
    
    for client_id in range(total_clients):
        if args.eta_clustering:
            cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"eta_{client_id}_of_{total_clients}_normal_l2.pkl"
        else:
            cluster_path = PROJECT_DIR / "data" / args.dataset / "clusters" / f"{client_id}_of_{total_clients}_normal_l2.pkl"
        with open(cluster_path, "rb") as f:
                normal_list = pickle.load(f)   
        all_normal.update(normal_list)


    # new multivariate normal distribution
    from scipy.stats import multivariate_normal
    
    def sum_multivariate_normals(x):
        total_sample_length = 0
        for _, _, client_sample_length, _  in gmm_params.values():
            total_sample_length += client_sample_length
        total_coff = 0
        final_probs = 0
        for i in range(total_clients):
            mu_GMM, var_GMM ,  client_sample_length, _ = gmm_params[i]
            p_client = multivariate_normal(mu_GMM, var_GMM)
            probs = p_client.pdf(x)
            coff = client_sample_length / total_sample_length
            total_coff += coff

            final_probs += probs * coff

        # print("total_coff", total_coff)
        return final_probs


    # new_p = sum_multivariate_normals(normal_params)


    ground_truth = {} 
    length = 0.2 
    for (idel, sample_matrix) in all_abnormal.items(): 

        # feature extraction 
        # sample_matrix = np.sum(np.square(sample), axis=1)  # for just l2
        # sample_matrix = get_matrix(sample)
        
        # get p values
        probs = sum_multivariate_normals(sample_matrix)
        temp_list = []
        temp_list += [0.0] * len(probs)
        
        window_size = int(len(probs) * length)  # fixed
        temp = []
        for idx in range(0, len(probs) - window_size + 1):
            arr = 0
            for i in range(idx, idx + window_size - 1):
                arr += abs(probs[i+1] - probs[i])
            temp.append(arr)
            

        for i in range(temp.index(max(temp)), temp.index(max(temp)) + window_size):
            temp_list[i] = 1.0

        ground_truth[idel] = temp_list



    final_gt = {}

    for i in all_normal.keys():
        final_gt[i] = [0.0] * len(all_normal[i])

    for i in all_abnormal.keys():
        final_gt[i] = ground_truth[i]   

    pl = {}
 
    pl_idx = list(range(vids_num))
    for i in pl_idx :
        pl[i] = None

    for k in final_gt.keys():
        pl[k] = final_gt[k]

    pl_array = [pls for pls in pl.values()]
    flattened = []
    for i in pl_array:
        flattened += i
     
    
    return np.array(flattened)





class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self,args, transform=None, test_mode=False):
        try:
            partition_path1 = PROJECT_DIR / "data" / args.dataset / args.partition
            partition_path2 = PROJECT_DIR / "data" / args.dataset / args.video_num_partition
            partition_path3 = PROJECT_DIR / "data" / args.dataset / args.partition_chain
            with open(partition_path1, "rb") as f:
                self.partition = pickle.load(f)
            with open(partition_path2, "rb") as f:
                self.video_num_partition = pickle.load(f)
            with open(partition_path3, "rb") as f:
                self.partition_chain = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        if args.dataset == 'ucf':
            self.no_of_vids = 1317 #1608 
        else:
            self.no_of_vids = 3954

        if test_mode:
            if args.dataset == 'ucf':
                self.con_all = np.load("data/datasets/Concat_test_10.npy")
            if args.dataset == 'XD':
                self.con_all = np.load("data/datasets/concat_XD_test.npy")
            print('self.con_all shape:',self.con_all.shape)
        else:
            
            
            if args.dataset == 'ucf':
                self.con_all = np.load('data/datasets/concat_UCF_V2.npy')[85902:]
            if args.dataset == 'XD':
                self.con_all = np.load("data/datasets/concat_XD.npy")    
            print('self.con_all shape:',self.con_all.shape)
            if args.train_mode == 'WS':
                self.label_ab = np.load('PL/Pseudo_Lebels_Maximum_AUC_0.8464948944321682.npy')
                self.label_all = np.concatenate((np.zeros((779951-len(self.label_ab),)), self.label_ab ), axis=0)
            else:
                
                gmm_p = self.all_pl(args)

                # self.label_ab = np.load('C:/Users/User/PycharmProjects/FL_AD/Pseudo_Lebels_Maximum_AUC_0.8464948944321682.npy')
                # self.label_ws = np.concatenate((np.zeros((779951-len(self.label_ab),)), self.label_ab ), axis=0)
                if args.gmm_pl == 1:
                    self.label_all = gmm_PL(args, self.partition['separation']['total'], gmm_p, self.no_of_vids)
                    # print("WS percentage:", args.ws_percentage/self.partition['separation']['total'])
                    # for k in range(args.ws_percentage):
                        
                    #     for i in self.partition_chain['data_indices'][k]['train']:
                    #         self.label_all[i] =  self.label_ws[i]

            print('self.label_all shape:',self.label_all.shape)

        self.tranform = transform
        self.test_mode = test_mode

    def all_pl(self, args):
        df = pd.DataFrame(columns=['num_of_GT_normal', 'num_of_GT_abnormal', 'num_of_P_normal', 'num_of_P_abnormal', 'Total_vids', 'correct_normal', 'correct_abnormal', 'normal_acc %', 'abnormal_acc %'])

        cluster_df = pd.DataFrame(columns=['Video_ID', 'Class', 'max_parameter', 'mean_parameter'])
        total_clients = self.partition['separation']['total']

        pl = {}
        gmm_params = {}

        pl_idx = list(range(self.no_of_vids))
        for i in pl_idx :
            pl[i] = None

        for i in range(total_clients):
            if args.eta_clustering == 1:
                client_pl, df, params = C2FPL_client_eta(self.con_all,args,  self.partition['data_indices'][i], self.video_num_partition['data_indices'][i], i, total_clients, df, args.load)
                gmm_params[i] = params
            else:
                client_pl, df, params = C2FPL_client(self.con_all,args,  self.partition['data_indices'][i], self.video_num_partition['data_indices'][i], i, total_clients, df, args.load)
                gmm_params[i] = params
        #     for k in client_pl.keys():
        #         pl[k] = client_pl[k]
        # pl_array = [pls for pls in pl.values()]
        # flattened = []
        # for i in pl_array:
        #     flattened += i
        # self.label_all = np.array(flattened)
        # self.label_all = np.load("original_pl.npy")




        # np.save("original_pl.npy",self.label_all )
        with open(f"data/ucf/clusters/gmm_params.pkl", "wb") as f:
            pickle.dump(gmm_params, f)
        # gmm_path = "data/ucf/clusters/gmm_params.pkl"
        # with open(gmm_path, "rb") as f:
        #     gmm_params = pickle.load(f)
        return gmm_params

        # # save pd to csv
        # cluster_df.to_csv('C:/Users/User/PycharmProjects/FL_AD/Clusters.csv', index=False)
        # wandb.log({f"Cluster_status": wandb.Table(dataframe=cluster_df)})
        # df.to_csv('C:/Users/User/PycharmProjects/FL_AD/Clustering_ACC.csv', index=False)
        # wandb.log({f"Clustering Accuarcy": wandb.Table(dataframe=cluster_df)})       

        

    def __getitem__(self, index):

        
        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
            return features
        else:
            
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
            labels = np.array(self.label_all[index], dtype=np.float32)

            return features , labels , index
    def __len__(self):
        return len(self.label_all)






DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "emnist": EMNIST,
    "fmnist": FashionMNIST,
    "femnist": FEMNIST,
    "medmnistS": MedMNIST,
    "medmnistC": MedMNIST,
    "medmnistA": MedMNIST,
    "covid19": COVID19,
    "celeba": CelebA,
    "synthetic": Synthetic,
    "svhn": SVHN,
    "usps": USPS,
    "tiny_imagenet": TinyImagenet,
    "cinic10": CINIC10,
    "domain": DomainNet,
    "ucf": Dataset_Con_all_feedback_XD,
    "XD": Dataset_Con_all_feedback_XD,
}

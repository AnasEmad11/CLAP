import json
from collections import OrderedDict
from typing import Dict, List, Optional, Type
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as torch_init
from .utils import PROJECT_DIR

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: torch.Tensor):
        return self.classifier(F.relu(self.base(x)))

    def get_final_features(self, x: torch.Tensor, detach=True) -> torch.Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "medmnistS": (1, 1024, 11),
            "medmnistC": (1, 1024, 11),
            "medmnistA": (1, 1024, 11),
            "covid19": (3, 196736, 4),
            "fmnist": (1, 1024, 10),
            "emnist": (1, 1024, 62),
            "femnist": (1, 1, 62),
            "cifar10": (3, 1600, 10),
            "cinic10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
            "tiny_imagenet": (3, 3200, 200),
            "celeba": (3, 133824, 2),
            "svhn": (3, 1600, 10),
            "usps": (1, 800, 10),
            "domain": infer(dataset, "avgcnn"),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 512),
            )
        )
        self.classifier = nn.Linear(512, config[dataset][2])


class LeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        config = {
            "mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "cinic10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
            "domain": infer(dataset, "lenet5"),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
            )
        )

        self.classifier = nn.Linear(84, config[dataset][2])


class TwoNN(DecoupledModel):
    def __init__(self, dataset):
        super(TwoNN, self).__init__()
        config = {
            "mnist": (784, 10),
            "medmnistS": (784, 11),
            "medmnistC": (784, 11),
            "medmnistA": (784, 11),
            "fmnist": (784, 10),
            "emnist": (784, 62),
            "femnist": (784, 62),
            "cifar10": (3072, 10),
            "cinic10": (3072, 10),
            "svhn": (3072, 10),
            "cifar100": (3072, 100),
            "usps": (1536, 10),
            "synthetic": (60, 10),  # default dimension and classes
        }

        self.base = nn.Linear(config[dataset][0], 200)
        self.classifier = nn.Linear(200, config[dataset][1])
        self.activation = nn.ReLU()

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.base(x))
        x = self.classifier(x)
        return x

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class MobileNetV2(DecoupledModel):
    def __init__(self, dataset):
        super(MobileNetV2, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "mobile"),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[1].in_features, config[dataset]
        )

        self.base.classifier[1] = nn.Identity()


class ResNet18(DecoupledModel):
    def __init__(self, dataset):
        super(ResNet18, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "res18"),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(self.base.fc.in_features, config[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "covid19": 4,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "alex"),
        }
        if dataset not in config.keys():
            raise NotImplementedError(f"AlexNet does not support dataset {dataset}")

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, config[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class SqueezeNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "sqz"),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        sqz = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        )
        self.base = sqz.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(sqz.classifier[1].in_channels, config[dataset], kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return self.classifier(self.base(x))

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


# class C2FPL(nn.Module):
#     def __init__(self, dataset):
#         super(C2FPL, self).__init__()
#         config = {
#             "mnist": 10,
#             "medmnistS": 11,
#             "medmnistC": 11,
#             "medmnistA": 11,
#             "fmnist": 10,
#             "svhn": 10,
#             "emnist": 62,
#             "femnist": 62,
#             "cifar10": 10,
#             "cinic10": 10,
#             "cifar100": 100,
#             "covid19": 4,
#             "usps": 10,
#             "celeba": 2,
#             "tiny_imagenet": 200,
            
#         }

#         self.fc1 = nn.Linear(1024*3, 512)
#         self.fc2 = nn.Linear(512, 32)
#         self.fc3 = nn.Linear(32, config[dataset])
#         self.dropout = nn.Dropout(0.6)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.apply(weight_init)

#     def forward(self, inputs):

#         x = torch.flatten(inputs, start_dim=1)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.sigmoid(self.fc3(x))
        
#         return x

class C2FPL_ucf(nn.Module): # multiplication then Addition
    def __init__(self, dataset):
        super(C2FPL_ucf, self).__init__()
        self.fc1 = nn.Linear(2048, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(2048, 512), nn.Softmax(dim = 1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim = 1))

        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)


    def forward(self, inputs):

        bs , ncrops, f = inputs.size()
        x = self.fc1(inputs)

   

        x = self.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
   

        x = self.relu(x)
        
        x = self.dropout(x)

        x = self.sigmoid(self.fc3(x))

        x = x.mean(dim = 1)

        return x

class C2FPL_XD(nn.Module): # multiplication then Addition
    def __init__(self, dataset):
        super(C2FPL_XD, self).__init__()
        self.fc1 = nn.Linear(1024, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(1024, 512), nn.Softmax(dim = 1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim = 1))

        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.apply(weight_init)


    def forward(self, inputs):

        bs , ncrops, f = inputs.size()
        x = self.fc1(inputs)

   

        x = self.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
   

        x = self.relu(x)
        
        x = self.dropout(x)

        x = self.sigmoid(self.fc3(x))

        x = x.mean(dim = 1)

        return x


# Some dirty codes for adapting DomainNet
def infer(dataset, model_type):
    if dataset == "domain":
        with open(PROJECT_DIR / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        class_num = metadata["class_num"]
        img_size = metadata["image_size"]
        coef = {"avgcnn": 50, "lenet5": 42.25}
        if model_type in ["alex", "res18", "sqz", "mobile"]:
            return class_num
        return (3, int(coef[model_type] * img_size), class_num)


MODEL_DICT: Dict[str, Type[DecoupledModel]] = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "2nn": TwoNN,
    "mobile": MobileNetV2,
    "res18": ResNet18,
    "alex": AlexNet,
    "sqz": SqueezeNet,
    "c2fpl_ucf": C2FPL_ucf,
    "c2fpl_XD": C2FPL_XD,
}

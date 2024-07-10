import pickle
from itertools import chain
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
# from dataset import Dataset
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.config.utils import trainable_params, get_best_device, evaluate, Logger
from src.config.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.args = args
        self.device = device
        self.client_id: int = None
        

        
        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / self.args.partition_chain
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        if self.args.dataset == "ucf" or self.args.dataset == "XD":
            general_data_transform = None

        else:
            general_data_transform = transforms.Compose(
                [transforms.Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
            )
            general_target_transform = transforms.Compose([])
            train_data_transform = transforms.Compose([])
            train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------
        if self.args.dataset == "ucf" or self.args.dataset == "XD":
                
            self.dataset_train = DATASETS[self.args.dataset](args, transform=None, test_mode=False
            )
            self.dataset_test = DATASETS[self.args.dataset](args, transform=None, test_mode=True
            )
            self.trainloader: DataLoader = None
            self.testloader: DataLoader = None
            self.trainset: Subset = Subset(self.dataset_train, indices=[])
            self.testset: Subset = Subset(self.dataset_test, indices=[])
            self.global_testset: Subset = None
            if self.args.global_testset:
                all_testdata_indices = []
                for indices in self.data_indices:
                    all_testdata_indices.extend(indices["test"])
                self.global_testset = Subset(self.dataset_test, sorted(all_testdata_indices))
        else:
            self.dataset = DATASETS[self.args.dataset](
                root=PROJECT_DIR / "data" / args.dataset,
                args=args.dataset_args,
                general_data_transform=general_data_transform,
                general_target_transform=general_target_transform,
                train_data_transform=train_data_transform,
                train_target_transform=train_target_transform,
            )

            self.trainloader: DataLoader = None
            self.testloader: DataLoader = None
            self.trainset: Subset = Subset(self.dataset, indices=[])
            self.testset: Subset = Subset(self.dataset, indices=[])
            self.global_testset: Subset = None
            if self.args.global_testset:
                all_testdata_indices = []
                for indices in self.data_indices:
                    all_testdata_indices.extend(indices["test"])
                self.global_testset = Subset(self.dataset, all_testdata_indices)
        ########################
        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        if self.args.dataset == "ucf" or self.args.dataset == "XD":
            self.criterion = torch.nn.BCELoss().to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}

        self.optimizer = torch.optim.SGD(
            params=trainable_params(self.model),
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10], gamma=0.1)
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def load_dataset(self, Refine=False):
        """This function is for loading data indices for No.`self.client_id` client."""
        # if self.args.train_mode == 'US':
        #     self.trainset.indices = list(chain(*self.data_indices[self.client_id]["train"]))
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size,shuffle=False) # we have both options for testing (one test for all clients or one test for each client)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size,shuffle=False) # this means that the test set is shared


    def train_and_log_ucf(self, Refine = False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        before = {
            "test_AUC": 0,
        }
        after = deepcopy(before)
        before = self.evaluate_ucf()
        if self.local_epoch > 0:
            update = self.fit_ucf(Refine)
            self.save_state()
            after = self.evaluate_ucf()

        eval_stats = {"before": before, "after": after}
        return eval_stats, update





    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            after = self.evaluate()
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                )

        eval_stats = {"before": before, "after": after}
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters from the same layers
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())


    def train_ucf(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
        Refine = False,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_stats, update  = self.train_and_log_ucf(Refine)  
        
        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1 

            return delta, len(self.trainset), eval_stats, update
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
            )












    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_stats = self.train_and_log(verbose=verbose) 

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit_ucf(self, Refine = False):
        # cluster_path = PROJECT_DIR / "data" / self.args.dataset / "clusters" / "gmm_params.pkl"
        # with open(cluster_path, "rb") as f:
        #         gmm_params = pickle.load(f)    
        # from scipy.stats import multivariate_normal
        # def sum_multivariate_normals(x):
        #     total_sample_length = 0
        #     for _, _, client_sample_length, _  in gmm_params.values():
        #         total_sample_length += client_sample_length
        #     total_coff = 0
        #     final_probs = 0
        #     for i in range(len(gmm_params)):
        #         mu_GMM, var_GMM ,  client_sample_length, _ = gmm_params[i]
        #         p_client = multivariate_normal(mu_GMM, var_GMM)
        #         probs = p_client.pdf(x)
        #         coff = client_sample_length / total_sample_length
        #         total_coff += coff

        #         final_probs += probs * coff
        #     return final_probs
        # print("total_coff", total_coff)
            
        with torch.set_grad_enabled(True):
            self.model.train()
            global_params = trainable_params(self.model, detach=True)
            loss_fn = torch.nn.BCELoss()
            losses = []
            # if Refine:
            #     refined_pl = torch.tensor(np.load("refined_pl.npy"))
            #     print("Load refined labels")
                
            self.scores = {}
            for _ in range(self.local_epoch):
                for (input, labels, idx)  in self.trainloader:
                    
                    input, labels = input.to(self.device), labels.to(self.device)


                    # if Refine:
                    #     # print("Load refined labels")
                    #     labels = refined_pl[idx]
                    #     labels = labels.to(self.device)

                    labels = labels.float()

                    
                    scores = self.model(input)
                    scores = scores.float().flatten()
                    
                    
                    trans = scores
                    trans = trans.cpu().detach().numpy()                   
                
                    for i, v in enumerate(idx.cpu().detach().numpy()):
                        self.scores[v] = trans[i]
                    loss = loss_fn(scores, labels)
                    losses.append(loss.cpu().detach().numpy())
                    self.optimizer.zero_grad()



                    loss.backward()
                 
                    self.optimizer.step()

        wandb.log({"Train Loss":np.mean(losses)}) 

        return self.scores

         




    @torch.no_grad()


    def evaluate_ucf(self, model: torch.nn.Module = None) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.

        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        with torch.no_grad():
            eval_model = self.model if model is None else model
            eval_model.eval()
            pred = torch.zeros(0, device=self.device)

            for input in self.testloader:
                input = input.to(self.device)
                logits = eval_model(inputs=input)

                pred = torch.cat((pred, logits))
                # print(pred.shape)


            # gt = np.load("C:/Users/User/PycharmProjects/FL_AD/gt.npy")[:2329200] #XD
            gt = np.load("labels/gt-ucf-RTFM.npy")  
            pred = list(pred.cpu().detach().numpy())
            pred = np.repeat(np.array(pred), 16)
            gt = gt[:len(pred)] 

            fpr, tpr, threshold = roc_curve(list(gt), pred)
            

            rec_auc = auc(fpr, tpr)


            precision, recall, th = precision_recall_curve(list(gt), pred)

            pr_auc = auc(recall, precision)

            wandb.log({"AUC": rec_auc, "AP": pr_auc})
            
            return {"test_AUC":rec_auc}
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.

        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")


        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num = evaluate(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evaluate(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def test_ucf(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "test_AUC": 0,
 
        }
        after = deepcopy(before)

        before = self.evaluate_ucf()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate_ucf()
        wandb.log({"before": before, "after": after})
        return {"before": before, "after": after}


# watermark version

import logging
import os
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.vpt_backbone import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import count_parameters

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 1


def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.fs_lr = args["fs_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.midfeature = None

    def _get_classifier_stage_trainable_param_count(self):
        fc = self._network.fc
        active_out = fc.out_features if fc.old_out == 0 else fc.out_features - fc.old_out
        sigma_params = 1 if getattr(fc, "sigma", None) is not None else 0
        return active_out * fc.in_features + sigma_params

    def get_stage_trainable_params_m(self):
        prompt_params = getattr(self, "task_trainable_params", 0)
        classifier_params = self._get_classifier_stage_trainable_param_count()
        return (prompt_params + classifier_params) / 1e6

    def get_final_trainable_params_m(self):
        tsp_total = (
            self._network.backbone.TSP.prompt_num
            * (self._network.backbone.TSP.e_p_length + 2)
            * self._network.backbone.TSP.emb_d
            * len(self._network.backbone.TSP.e_layers)
        )
        rsp_total = 0
        if hasattr(self._network.backbone, "RSP"):
            rsp_total = (
                self._network.backbone.RSP.e_pool_size
                * (self._network.backbone.RSP.e_p_length + 2)
                * self._network.backbone.RSP.emb_d
                * len(self._network.backbone.RSP.e_layers)
            )
        fc = self._network.fc
        classifier_total = fc.out_features * fc.in_features + (1 if getattr(fc, "sigma", None) is not None else 0)
        return (tsp_total + rsp_total + classifier_total) / 1e6

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        # # use class prototype as classifier weights.
        # model = model.eval()
        # embedding_list = []
        # label_list = []
        # if self._cur_task > 0:
        #     with torch.no_grad():
        #         for i, batch in enumerate(trainloader):
        #             (_, data, label) = batch
        #             data = data.to(self._device)
        #             label = label.to(self._device)
        #             embedding = model(data)["features"]
        #             embedding_list.append(embedding.cpu())
        #             label = label.repeat(int(embedding.shape[0] / label.shape[0]))
        #             label_list.append(label.cpu())
        #     embedding_list = torch.cat(embedding_list, dim=0)
        #     label_list = torch.cat(label_list, dim=0)
        # else:
        #     with torch.no_grad():
        #         for i, batch in enumerate(trainloader):
        #             (_, data, label) = batch
        #             data = data.to(self._device)
        #             label = label.to(self._device)
        #             embedding = model(data)["features"]
        #             embedding_list.append(embedding.cpu())
        #             label_list.append(label.cpu())
        #     embedding_list = torch.cat(embedding_list, dim=0)
        #     label_list = torch.cat(label_list, dim=0)
        #
        # class_list = np.unique(trainloader.dataset.labels)
        # for class_index in class_list:
        #     data_index = (label_list == class_index).nonzero().squeeze(-1)
        #     embedding = embedding_list[data_index]
        #     proto = embedding.mean(0)
        #     self._network.fc.weight.data[class_index] = proto
        # use class prototype as classifier weights.
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(trainloader.dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # print(self._total_classes)
        self._network.update_fc(self._total_classes)
        # to evaluate the performance on future classes
        self._network.backbone.TSP.process_task_count(self._total_classes)
        self._network.backbone.RSP.process_task_count()

        # 统计实际参与训练的参数量（考虑detach的影响）
        # 必须在process_task_count()之后，此时TSP已经初始化
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        
        # 计算TSP实际训练的prompt数量
        if self._cur_task == 0:
            # Task 0: 所有prompts都参与训练
            tsp_active_prompts = self._network.backbone.TSP.prompt_num
            # RSP使用e_pool_size，且Task 0全部参与训练
            rsp_active_prompts = self._network.backbone.RSP.e_pool_size if hasattr(self._network.backbone, 'RSP') else 0
        else:
            # Task 1+: 只有新增的prompts参与训练（旧的被detach）
            tsp_active_prompts = self._network.backbone.TSP.prompt_num - self._network.backbone.TSP.last_prompt_num
            rsp_active_prompts = 0  # RSP在增量阶段被冻结
        
        # 计算实际训练参数量
        # TSP: active_prompts × (e_p_length + 1 + 1) × embed_dim × num_layers
        # e_p_length个token + 1个key + 1个attention
        tsp_params_per_layer = tsp_active_prompts * (self._network.backbone.TSP.e_p_length + 2) * self._network.backbone.TSP.emb_d
        tsp_total_params = tsp_params_per_layer * len(self._network.backbone.TSP.e_layers)
        
        # RSP: 同样的计算方式
        if rsp_active_prompts > 0:
            rsp_params_per_layer = rsp_active_prompts * (self._network.backbone.RSP.e_p_length + 2) * self._network.backbone.RSP.emb_d
            rsp_total_params = rsp_params_per_layer * len(self._network.backbone.RSP.e_layers)
        else:
            rsp_total_params = 0
        
        actual_trainable_params = tsp_total_params + rsp_total_params
        
        logging.info("All params: {}".format(total_params))
        logging.info("Actual trainable params (excluding detached): {}".format(actual_trainable_params))
        logging.info("TSP active prompts: {} (total: {})".format(tsp_active_prompts, self._network.backbone.TSP.prompt_num))
        if hasattr(self._network.backbone, 'RSP'):
            logging.info("RSP active prompts: {} (total: {})".format(rsp_active_prompts, self._network.backbone.RSP.e_pool_size))
        logging.info("Excluded params: head, detached prompts")
        
        # 详细打印每层的参数量
        if actual_trainable_params > 0:
            # TSP参数详情
            tsp_e_p_params = tsp_active_prompts * self._network.backbone.TSP.e_p_length * self._network.backbone.TSP.emb_d
            tsp_e_k_params = tsp_active_prompts * self._network.backbone.TSP.emb_d
            tsp_e_a_params = tsp_active_prompts * self._network.backbone.TSP.emb_d
            
            for layer in self._network.backbone.TSP.e_layers:
                logging.info("backbone.TSP.e_p_{}: {}".format(layer, tsp_e_p_params))
                logging.info("backbone.TSP.e_k_{}: {}".format(layer, tsp_e_k_params))
                logging.info("backbone.TSP.e_a_{}: {}".format(layer, tsp_e_a_params))
            
            # RSP参数详情
            if rsp_active_prompts > 0:
                rsp_e_p_params = rsp_active_prompts * self._network.backbone.RSP.e_p_length * self._network.backbone.RSP.emb_d
                rsp_e_k_params = rsp_active_prompts * self._network.backbone.RSP.emb_d
                rsp_e_a_params = rsp_active_prompts * self._network.backbone.RSP.emb_d
                
                for layer in self._network.backbone.RSP.e_layers:
                    logging.info("backbone.RSP.e_p_{}: {}".format(layer, rsp_e_p_params))
                    logging.info("backbone.RSP.e_k_{}: {}".format(layer, rsp_e_k_params))
                    logging.info("backbone.RSP.e_a_{}: {}".format(layer, rsp_e_a_params))

        if isinstance(self.args["kshot"], int) and self._known_classes > 0:
            train_bs = self.args["fs_batch_size"]
        else:
            train_bs = self.batch_size
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            kshot=self.args["kshot"],
        )
        self.train_dataset = train_dataset

        self.train_loader = DataLoader(
            train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers
        )

        self.data_manager = data_manager

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        if self._total_classes < self.args['nb_classes']:
            future_indices = np.arange(self._total_classes, self.args["nb_classes"])
            self.future_train_dataset = data_manager.get_dataset(
                future_indices,
                source="train",
                mode="test",
                kshot=self.args["kshot"],
            )
            self.future_train_loader = DataLoader(
                self.future_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.future_test_dataset = data_manager.get_dataset(
                future_indices,
                source="test",
                mode="test",
            )
            self.future_test_loader = DataLoader(
                self.future_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.future_dataset = self.future_test_dataset
            self.future_loader = self.future_test_loader
        test_curr_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="test",
            mode="test",
        )

        self.test_curr_loader = DataLoader(
            test_curr_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
            kshot=self.args["kshot"],
        )

        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=int(self.batch_size / 2),
            shuffle=False,
            num_workers=num_workers,
        )

        logging.info(
            "training set size: {}, fc construct set size: {}".format(
                len(train_dataset), len(train_dataset_for_protonet)
            )
        )

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)
        if self._cur_task > 0:
            self._network.backbone.Freeze_new()

        if os.path.exists(self.args["base_model_path"]) and self._cur_task == 0:
            logging.info(
                "================= load base model from: {} =================".format(
                    self.args["base_model_path"]
                )
            )
            # print(self._cur_task)
            self._network.load_state_dict(torch.load(self.args["base_model_path"]))
            # self.replace_midfeature(train_loader_for_protonet, self._network, None)
            # self.replace_fc(train_loader_for_protonet, self._network, None)

        else:
            if self._cur_task > 0:
                self.replace_fc(train_loader_for_protonet, self._network, None)

            if self._cur_task == 0:
                if self.args["optimizer"] == "sgd":
                    optimizer = optim.SGD(
                        self._network.parameters(),
                        momentum=0.9,
                        lr=self.init_lr,
                        weight_decay=self.weight_decay,
                    )
                elif self.args["optimizer"] == "adam":
                    optimizer = optim.AdamW(
                        self._network.parameters(),
                        lr=self.init_lr,
                        weight_decay=self.weight_decay,
                    )
                
                # Record actual trainable parameters (excluding classifier and detached prompts)
                # 使用之前计算的实际训练参数量
                if self._cur_task == 0:
                    tsp_active_prompts = self._network.backbone.TSP.prompt_num
                    rsp_active_prompts = self._network.backbone.RSP.e_pool_size if hasattr(self._network.backbone, 'RSP') else 0
                else:
                    tsp_active_prompts = self._network.backbone.TSP.prompt_num - self._network.backbone.TSP.last_prompt_num
                    rsp_active_prompts = 0
                
                tsp_params_per_layer = tsp_active_prompts * (self._network.backbone.TSP.e_p_length + 2) * self._network.backbone.TSP.emb_d
                tsp_total_params = tsp_params_per_layer * len(self._network.backbone.TSP.e_layers)
                
                if rsp_active_prompts > 0:
                    rsp_params_per_layer = rsp_active_prompts * (self._network.backbone.RSP.e_p_length + 2) * self._network.backbone.RSP.emb_d
                    rsp_total_params = rsp_params_per_layer * len(self._network.backbone.RSP.e_layers)
                else:
                    rsp_total_params = 0
                
                self.task_trainable_params = tsp_total_params + rsp_total_params
                
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
                )
                self._init_train(
                    train_loader,
                    test_loader,
                    train_loader_for_protonet,
                    optimizer,
                    scheduler,
                )
            else:
                if self.args["optimizer"] == "sgd":
                    optimizer = optim.SGD(
                        self._network.parameters(),
                        momentum=0.9,
                        lr=self.fs_lr,
                        weight_decay=self.weight_decay,
                    )
                elif self.args["optimizer"] == "adam":
                    optimizer = optim.AdamW(
                        self._network.parameters(),
                        lr=self.fs_lr,
                        weight_decay=self.weight_decay,
                    )
                
                # Record actual trainable parameters (excluding classifier and detached prompts)
                # 使用之前计算的实际训练参数量
                if self._cur_task == 0:
                    tsp_active_prompts = self._network.backbone.TSP.prompt_num
                    rsp_active_prompts = self._network.backbone.RSP.e_pool_size if hasattr(self._network.backbone, 'RSP') else 0
                else:
                    tsp_active_prompts = self._network.backbone.TSP.prompt_num - self._network.backbone.TSP.last_prompt_num
                    rsp_active_prompts = 0
                
                tsp_params_per_layer = tsp_active_prompts * (self._network.backbone.TSP.e_p_length + 2) * self._network.backbone.TSP.emb_d
                tsp_total_params = tsp_params_per_layer * len(self._network.backbone.TSP.e_layers)
                
                if rsp_active_prompts > 0:
                    rsp_params_per_layer = rsp_active_prompts * (self._network.backbone.RSP.e_p_length + 2) * self._network.backbone.RSP.emb_d
                    rsp_total_params = rsp_params_per_layer * len(self._network.backbone.RSP.e_layers)
                else:
                    rsp_total_params = 0
                
                self.task_trainable_params = tsp_total_params + rsp_total_params
                
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args["fs_epoch"], eta_min=self.min_lr
                )
                self._init_train(
                    train_loader,
                    test_loader,
                    train_loader_for_protonet,
                    optimizer,
                    scheduler,
                )
            self.replace_fc(train_loader_for_protonet, self._network, None)

    def eval_task(self):
        y_pred, y_true = [], []
        prompt_time = self._eval_cnn(self.test_loader, y_pred, y_true)
        y_pred, y_true = self._eval_future_cnn(self.future_test_loader if hasattr(self, "future_test_loader") else self.future_loader, y_pred, y_true)
        accy = self._evaluate(y_pred, y_true)
        accy["prompt_time"] = prompt_time
        return accy

    def _eval_cnn(self, loader, y_pred, y_true):
        self._network.eval()

        # all_outputs, all_embedding = [], []
        prompt_time = 0.0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network(inputs)
                outputs = out["logits"][:, : self._total_classes]
                # embedding = out["features"]
                prompt_time += out.get("prompt_time", 0)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            # all_outputs.append(outputs.cpu())
            # all_embedding.append(embedding.cpu())

        # y_pred = np.concatenate(y_pred)
        # y_true = np.concatenate(y_true)
        # all_outputs = torch.cat(all_outputs)
        # all_embedding = torch.cat(all_embedding)

        return prompt_time  # [N, topk]

    def _eval_future_cnn(self, loader, y_pred, y_true):
        if self._total_classes < self.args['nb_classes']:
            future_train_loader = self.future_train_loader if hasattr(self, "future_train_loader") else self.future_loader
            if future_train_loader is None or loader is None:
                return np.concatenate(y_pred), np.concatenate(y_true)

            self.update_future_head(self._network, future_train_loader)
            for _, (_, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    # out = self._network(inputs)
                    # outputs = out["future_logits"]["logits"]
                    features = self._network(inputs)["features"]
                    outputs = self._network.future_head(features)["logits"]
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[
                    1
                ]  # [bs, topk]
                y_pred.append(predicts.cpu().numpy())  # Adjust for future tasks
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def update_future_head(self, model, future_loader):
        for i in range(self._total_classes):
            model.future_head.weight.data[i] = model.fc.weight.data[i]
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(future_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(future_loader.dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            model.future_head.weight.data[class_index] = proto

    # naive train
    def _init_train(
            self, train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler
    ):
        if isinstance(self.args["kshot"], int) and self._known_classes > 0:
            total_epoch = self.args["fs_epoch"]
        else:
            total_epoch = self.args["tuned_epoch"]
        for _, epoch in enumerate(range(total_epoch)):
            self._network.train()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs.shape)
                if self._cur_task == 0:

                    out = self._network(inputs, targets=targets)
                    logits = out["logits"]
                    loss_pc = out["loss_match"]
                    loss = (
                            F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]
                    )

                else:

                    out = self._network(inputs, train=True, targets=targets)
                    logits = out["logits"]
                    loss_pc = out["loss_match"]
                    targets = targets.repeat(int(logits.shape[0] / targets.shape[0]))
                    loss = (
                            F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]
                    )
                    # print(loss_pc)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            scheduler.step()

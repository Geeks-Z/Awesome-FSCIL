import logging
import os
import numpy as np
import torch
from torch import nn
import copy
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.px_d1_backbone import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_manager import pil_loader
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.distributions.multivariate_normal import MultivariateNormal
num_workers = 4

def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size= args["batch_size"]
        self.init_lr=args["init_lr"]
        self.fs_lr=args["fs_lr"]

        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        margin = float(self.args['common_param'])
        assert margin != 0, "margin should not equal 0"
        if margin > 0:
            self.m1, self.m2 = 0, margin
        else:
            self.m1, self.m2 = margin, -margin

        self.sigma_value = float(self.args['common_param2'])
        self.few_shot_protos = None
        self.few_shot_covs = None
        self.few_shot_sim_mats = None

    def after_task(self):
        self._known_classes = self._total_classes
    
    def update_proxy_fc(self):
        self.proxy_fc = self.generate_fc(self.out_dim, self.c.cur_task_size).to(self.device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.debug = False
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self._cur_task ==0:
            self._network.update_proxy_fc(self._total_classes - self._known_classes, sigma_value=self.sigma_value, m1=self.m1, m2=self.m2)
            self._network.update_fc(self._total_classes, sigma_value=self.sigma_value, m=0.)
            self.few_shot_margin = (self.m2 * torch.ones(self._total_classes)).to(self._device)
        else: # 增量阶段只使用self._network.fc
            weight = torch.zeros(self._total_classes - self._known_classes, self._network.feature_dim).to(self._device)
            self._network.fc.few_shot_weight  = nn.Parameter(weight)
            self.few_shot_margin = torch.cat([self.few_shot_margin,(self.m2 * torch.ones(self._total_classes-self._known_classes)).to(self._device)], dim=0)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", kshot=self.args["kshot"] )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            train_bs = self.args['kshot'] * (self._total_classes - self._known_classes)
        else:
            train_bs = self.batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        test_curr_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test" )
        self.test_curr_loader = DataLoader(test_curr_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", kshot=self.args["kshot"])
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=train_bs, shuffle=False, num_workers=num_workers)
        
        logging.info("training set size: {}, fc construct set size: {}".format(len(train_dataset), len(train_dataset_for_protonet)))

        if self.args["base_model_path"]:
            if self._cur_task == 0:
                special_model = self.args["base_model_path_init"][:-4]+'_special.pth'
                if not os.path.exists(special_model):
                    self._network.load_state_dict(torch.load(self.args["base_model_path_init"]))
                    self._network.to(self._device)
                    self.calc_fish_matrix(dataset=train_dataset_for_protonet, data_loader=self.train_loader_for_protonet,
                                          debug=self.debug)
                    if self.args.get("save_checkpoints", True):
                        torch.save(
                            self._network.state_dict(),
                            os.path.join(self.args["saved_path"], "session_0_special.pth"),
                        )
                else:
                    self._network.backbone.generator_combined_weight()
                    self._network.load_state_dict(torch.load(special_model))
            if self._cur_task == self.args['nb_tasks'] - 1:
                assert os.path.exists(self.args["base_model_path"])
                logging.info('================= load base model from: {} ================='.format(self.args["base_model_path"]))
                self._network.fc = None
                self._network.update_fc(self._total_classes, sigma_value=self.sigma_value, m=0.)
                self._network.to(self._device)
                self._network.eval()
                load_msg = self._network.load_state_dict(torch.load(self.args["base_model_path"]),strict=False)
                logging.info(load_msg)
            return

        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        # replace fc
        if self._cur_task == 0:
            self.calc_fish_matrix(dataset=train_dataset_for_protonet, data_loader=self.train_loader_for_protonet,debug=self.debug)
            self._network.replace_fc(dataset=train_dataset_for_protonet, data_loader=self.train_loader_for_protonet)
            # self.base_proxy_fc_weight = copy.deepcopy(self._network.proxy_fc.weight.data)
            # self.base_proxy_fc2_weight = copy.deepcopy(self._network.proxy_fc2.weight.data)
            # self.combined_base_protos = copy.deepcopy((self._network.proxy_fc.weight.data + self._network.proxy_fc2.weight.data) * 0.5)
            self.combined_base_protos = copy.deepcopy(self._network.proxy_fc2.weight.data)

            if self.args['special_flag']:  # 替换fc
                # self._network.fc.weight.data = copy.deepcopy(
                #     (self._network.proxy_fc.weight.data + self._network.proxy_fc2.weight.data) * 0.5).to(self._device)
                # 计算fisher矩阵
                # self.calc_fish_matrix(dataset=train_dataset_for_protonet, data_loader=self.train_loader_for_protonet, debug=self.debug)
                logging.info('-------Execute special program: using merged proxy fc as final fc of old class --------')

    def scape_train(self, data_manager, scale_1, scale_2):
        self._network.fc = self._network.generate_fc(self._network.feature_dim, data_manager.nb_classes, sigma=True,
                                                     sigma_value=16.0, m=0.).to(self._device)
        for cur_task in range(data_manager.nb_tasks):
            known_classes = sum([data_manager.get_task_size(_i) for _i in range(cur_task)]) if cur_task > 0 else 0
            total_classes = known_classes + data_manager.get_task_size(cur_task)
            logging.info("Learning on {}-{}".format(known_classes, total_classes))
            # --------> 准备数据
            train_dataset = data_manager.get_dataset(np.arange(known_classes, total_classes), source="train", mode="train", kshot=self.args["kshot"]) # 对齐随机性
            if isinstance(self.args['kshot'], int) and known_classes > 0:
                train_bs = self.args['kshot'] * (total_classes - known_classes)
            else:
                train_bs = self.batch_size
            # train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
            train_dataset_for_protonet = data_manager.get_dataset(np.arange(known_classes, total_classes),source="train", mode="test", kshot=self.args["kshot"])
            self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=train_bs, shuffle=False,num_workers=num_workers)

            scape_params = {"scale_1":scale_1, "scale_2":scale_2}
            # --------> 计算原型
            if cur_task ==0:
                with torch.no_grad():
                    prog_bar = tqdm(self.train_loader_for_protonet)
                    # extract embeddings
                    embedding_list, label_list = [], []
                    for _, batch in enumerate(prog_bar):
                        (_, data, label) = batch
                        data = data.to(self._device)
                        label = label.to(self._device)
                        embedding = self._network.forward_scape(data,scape_params,only_features=True)
                        embedding_list.append(embedding.cpu())
                        label_list.append(label.cpu())

                        prog_bar.set_description('Extracting prototype from PTM =>')

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                # construct prototype-based classifier
                class_list = sorted(np.unique(train_dataset_for_protonet.labels))
                proto_list = []
                for class_index in class_list:
                    # calculate prototype
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    proto_list.append(proto[None])
                protos_current = torch.cat(proto_list, dim=0)
            else:
                with torch.no_grad():
                    for i, (_, inputs, targets) in enumerate(self.train_loader_for_protonet):
                        inputs, targets = inputs.to(self._device), targets.to(self._device)
                    out = self._network.forward_scape(inputs,scape_params,only_features=True)
                embedding_list = out

                proto_list = []
                for cid in sorted(torch.unique(targets)):
                    data_index = (cid == targets).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    proto_list.append(copy.deepcopy(proto[None]))
                protos_current = torch.cat(proto_list, dim=0)
            self._network.fc.weight.data[known_classes:total_classes, :] = copy.deepcopy(protos_current)
        # test for final model
        test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        accy = self.eval_acc_scape(self.test_loader, scape_params)
        return accy


    def eval_acc_scape(self, loader, scape_params):
        y_pred, y_true = [], []
        all_outputs = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network.forward_scape(inputs,scape_params)
                outputs = out["logits"]
                # embedding = out["features"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        accy = self._evaluate(y_pred, y_true)
        return accy

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)

        # *********************** 设定可更新参数 **********************
        if self._cur_task == 0:
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if "proxy_fc" in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k_2" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v_2" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k_2" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v_2" + "." + str(self._cur_task) + "." in name:
                    param.requires_grad_(True)
                if "sigma" in name:
                    param.requires_grad_(False)
        else:
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if "few_shot_weight" in name:
                    param.requires_grad_(True)
                if "fc.weight" in name:
                    param.requires_grad_(True)
                if "proxy_fc" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")

        # *********************** 载入基础模型 **********************
        if self.args["base_model_path"] and self._cur_task==0:
            assert os.path.exists(self.args["base_model_path"])
            logging.info('================= load base model from: {} ================='.format(self.args["base_model_path"]))
            self._network.load_state_dict(torch.load(self.args["base_model_path"]))
            return

        if self._cur_task==0: #-------- 初始任务优化器-------
            params = [p for n, p in self._network.named_parameters() if p.requires_grad == True]
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        else: #-------- 小样本任务优化器-------
            params = [p for n, p in self._network.named_parameters() if p.requires_grad == True]
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(params, momentum=0.9, lr=self.fs_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(params, lr=self.fs_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['fs_epoch'], eta_min=self.min_lr)

        self._init_train(train_loader, test_loader, optimizer, scheduler,self.debug)
        if self.args.get("save_checkpoints", True):
            torch.save(
                self._network.state_dict(),
                os.path.join(
                    self.args["saved_path"],
                    "session_{}.pth".format(self._cur_task),
                ),
            )
        else:
            logging.info("Checkpoint saving is disabled for this experiment.")

        # if self._cur_task == 0:
        #     # self.update_ema_prompt(train_loader_for_protonet, mode='base')
        #     self.replace_fc(train_loader_for_protonet, self._network, None)

    def calibrate_analysis(self):
        y_pred, y_true = self._eval_acc(self.test_loader)
        top1_pred = y_pred.T[0]
        y_pred_binary = np.where(top1_pred < self.args['init_cls'], 1, 0)
        y_true_binary = np.where(y_true < self.args['init_cls'], 1, 0)
        # 计算混淆矩阵元素
        TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

        FNR = FN / (TP + FN) * 100
        FPR = FP / (FP + TN) * 100
        log_tmp = 'TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}'.format(TP, TN, FP, FN, FNR, FPR)
        logging.info(log_tmp)


    def eval_task(self):
        y_pred, y_true = self._eval_acc(self.test_loader)
        accy = self._evaluate(y_pred, y_true)
        return accy
    
    def _eval_acc(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        all_outputs, all_embedding = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network(inputs,test=True)
                outputs = out["logits"]
                embedding = out["features"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
            all_embedding.append(embedding.cpu())
            
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        all_outputs = torch.cat(all_outputs)
        all_embedding = torch.cat(all_embedding)

        return y_pred, y_true # [N, topk]

    def calc_fish_matrix(self, dataset, data_loader, debug):
        self._network.eval()
        real_m1 = self._network.proxy_fc.m
        real_m2 = self._network.proxy_fc2.m
        data_loader_len = 10 if debug else len(data_loader)
        self._network.backbone.generator_combined_weight()
        # ********************** 计算第一个模型的fish信息 **************************
        logging.info("Calculate the fish information of the first model")
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # if "proxy_fc" in name:
            #     param.requires_grad_(True)
            if "lora_weight_k_1" in name:
                param.requires_grad_(True)
            if "lora_weight_v_1" in name:
                param.requires_grad_(True)

        # Double check
        enabled_1 = {}
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled_1[name] = param
        enabled_1 = {k: enabled_1[k] for k in sorted(enabled_1)}
        # ------------- 1-a
        fisher_a = {}
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                fisher_a[name] = torch.zeros_like(param.data)
        # self._network.proxy_fc.m = real_m2 # 切换边距，用于计算交叉梯度
        for i, (_, inputs, targets) in enumerate(data_loader):
            if debug and i > 10:
                break
            self._network.zero_grad()
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            # logits = self._network(inputs, label=targets, first_flow=91)["logits"]
            features = self._network.backbone(inputs, 0, first_flow=91)[:, 0]
            logits = self._network.proxy_fc(features, label=targets)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            for name, param in self._network.named_parameters():
                if param.grad is not None:
                    fisher_a[name] += param.grad.data ** 2
        # self._network.proxy_fc.m = real_m1 # 恢复边距
            # Normalize
        for name in fisher_a:
            fisher_a[name] /= data_loader_len

        # ********************** 计算第二个模型的fish信息 **************************
        logging.info("Calculate the fish information of the second model")

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # if "proxy_fc2" in name:
            #     param.requires_grad_(True)
            if "lora_weight_k_2" in name:
                param.requires_grad_(True)
            if "lora_weight_v_2" in name:
                param.requires_grad_(True)
            # if "sigma" in name:
            #     param.requires_grad_(False)
        # Double check
        enabled_2 = {}
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled_2[name] = param
        enabled_2 = {k: enabled_2[k] for k in sorted(enabled_2)}

        # # -------------- 2.1
        fisher_2a = {}
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                fisher_2a[name] = torch.zeros_like(param.data)
        self._network.proxy_fc2.m = real_m1  # 切换边距，用于计算交叉梯度
        for i, (_, inputs, targets) in enumerate(data_loader):
            if debug and i > 10:
                break
            self._network.zero_grad()
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            # logits = self._network(inputs, label=targets, first_flow=92)["logits"]
            features = self._network.backbone(inputs, 0, first_flow=92)[:, 0]
            logits = self._network.proxy_fc2(features, label=targets)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            for name, param in self._network.named_parameters():
                if param.grad is not None:
                    fisher_2a[name] += param.grad.data ** 2
            # Normalize
        self._network.proxy_fc2.m = real_m2  # 切换边距，用于计算交叉梯度
        for name in fisher_2a:
            fisher_2a[name] /= len(data_loader)
        # ********************** 合并fish 信息 **************************
        merged_state_dict = {}
        for key in fisher_a.keys():
            f1_a = fisher_a[key]
            f2_a = fisher_2a[key.replace('_1', '_2')]

            f1_a_norm = f1_a.norm(p=2, dim=(0, 1))
            f2_a_norm = f2_a.norm(p=2, dim=(0, 1))
            alpha = f1_a_norm / (f1_a_norm + f2_a_norm)
            beta = 1.0 - alpha
            logging.info("key : {} , f1 : {}, f2 : {} , alpha : {} , beta: {}".format(key,f1_a_norm.item(), f2_a_norm.item(), alpha.item(), beta.item()))

            param1 = enabled_1[key]
            param2 = enabled_2[key.replace('_1','_2')]
            # new_param = f1_norm * param2 + (1-f1_norm) * param1
            new_param = beta * param2 + alpha * param1
            merged_state_dict[key] = new_param

        for bid in range(len(self._network.backbone.blocks)):
            self._network.backbone.blocks[bid].attn.lora_weight_k_meg.data = copy.deepcopy(merged_state_dict[
                'backbone.blocks.{}.attn.lora_weight_k_1.weight'.format(bid)].detach())
            self._network.backbone.blocks[bid].attn.lora_weight_v_meg.data = copy.deepcopy(merged_state_dict[
                'backbone.blocks.{}.attn.lora_weight_v_1.weight'.format(bid)].detach())

        logging.info('finish the merge ')

    def update_few_shot_info(self, dataloader):
        self._network.eval()
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            out = self._network.backbone(inputs, 0)[:, 0]
        embedding_list = out

        proto_list = []
        for cid in sorted(torch.unique(targets)):
            data_index = (cid == targets).nonzero().squeeze(-1)

            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            proto_list.append(copy.deepcopy(proto[None]))
            # cov = torch.cov(embedding.T) + torch.eye(proto.shape[-1]) * 1e-3

        cur_protos = torch.cat(proto_list, dim=0)
        # self._network.fc.weight.data[-cur_protos.shape[0]:, :] = copy.deepcopy(cur_protos)
        self._network.fc.few_shot_weight.data = copy.deepcopy(F.normalize(cur_protos,p=2,dim=1)).to(self._device)

        base_protos = copy.deepcopy(self._network.protos_list_source[0]).to(self._device)
        base_covs = copy.deepcopy(self._network.covs_list_source[0]).to(self._device)

        cur_proto_norm = F.normalize(cur_protos, p=2, dim=1)  # [N, D]
        base_proto_norm = F.normalize(base_protos, p=2, dim=1)  # [M, D]
        sim_matrix = torch.matmul(cur_proto_norm, base_proto_norm.T) # [N, M]
        topk_sim_indices = sim_matrix.topk(k=1,largest=True)[1].squeeze()
        cur_covs = copy.deepcopy(base_covs[topk_sim_indices])
        # new_cur_protos = cur_protos - base_protos[topk_sim_indices] + self.base_proxy_fc_weight[topk_sim_indices]
        if self.few_shot_protos is None:
            self.few_shot_protos = copy.deepcopy(cur_protos).cpu()
            self.few_shot_covs = copy.deepcopy(cur_covs).cpu()
            # self.few_shot_sim_mats = copy.deepcopy(sim_matrix).cpu()
        else:
            self.few_shot_protos = torch.cat([self.few_shot_protos, copy.deepcopy(cur_protos).cpu()], dim=0)
            self.few_shot_covs = torch.cat([self.few_shot_covs, copy.deepcopy(cur_covs).cpu()], dim=0)
            # self.few_shot_sim_mats = torch.cat([self.few_shot_sim_mats, copy.deepcopy(sim_matrix).cpu()], dim=0)
        info = "Task {} => classifier is built".format(self._cur_task)
        logging.info(info)

    # naive train
    def _init_train(self, train_loader, test_loader, optimizer, scheduler, debug=False):
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            total_epoch = self.args['fs_epoch']
            self.update_few_shot_info(self.train_loader_for_protonet)  # 更新小样本原型，评估协方差
            self._network.eval()
            total_protos = copy.deepcopy(torch.cat([self._network.protos_list_source[0],self.few_shot_protos],dim=0))
            total_covs = copy.deepcopy(torch.cat([self._network.covs_list_source[0],self.few_shot_covs],dim=0))

            # *********************** 方式2 真实数据和采样数据混用 ****************************
            self.train_loader_for_protonet.dataset.trsf = self.train_dataset.trsf  # 切换到数据增强模式，用于训练
            for epoch in range(total_epoch):
                losses = 0.
                sampled_data = []
                sampled_label = []
                num_sampled_pcls = 256
                #  每个批次sample 新的数据
                # =========> 基础类别采样 <========
                for c_id in range(self.args['init_cls']):
                    # 标准原型采样
                    cls_mean = total_protos[c_id].to(self._device)
                    cls_cov = total_covs[c_id].to(self._device)  # 协方差使用原来计算的，不进行矫正
                    m = MultivariateNormal(cls_mean.float(), covariance_matrix=cls_cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    prob_density = m.log_prob(sampled_data_single)
                    cur_samples, index_prob = torch.topk(prob_density, num_sampled_pcls//2)
                    sampled_data.append(sampled_data_single[index_prob])
                    sampled_label.extend([c_id] * (num_sampled_pcls//2))
                    # 学习到的原型采样
                    cls_mean_c = self.combined_base_protos[c_id].to(self._device)
                    mc = MultivariateNormal(cls_mean_c.float(), covariance_matrix=cls_cov.float())
                    sampled_data_mc = mc.sample(sample_shape=(num_sampled_pcls,))
                    prob_density_mc = mc.log_prob(sampled_data_mc)
                    cur_samples_mc, index_prob_mc = torch.topk(prob_density_mc, num_sampled_pcls//2)
                    sampled_data.append(sampled_data_mc[index_prob_mc])
                    sampled_label.extend([c_id] * (num_sampled_pcls//2))
                # =========> 已经学习过得小样本类别采样 <========
                for c_id in range(self.args['init_cls'], self._known_classes):  # 遍历所有类别
                    cls_mean = total_protos[c_id].to(self._device)
                    cls_cov = total_covs[c_id].to(self._device)
                    m = MultivariateNormal(cls_mean.float(), covariance_matrix=cls_cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls*2,))
                    prob_density = m.log_prob(sampled_data_single)
                    cur_samples, index_prob = torch.topk(prob_density, num_sampled_pcls)
                    sampled_data.append(sampled_data_single[index_prob])
                    sampled_label.extend([c_id] * num_sampled_pcls)
                # =========> 当前任务小样本类别采样 <========
                if num_sampled_pcls % self.args['kshot'] == 0:
                    sample_num = num_sampled_pcls // self.args['kshot']
                else:
                    sample_num = num_sampled_pcls // self.args['kshot'] + 1
                for _ in range(sample_num):
                    with torch.no_grad():
                        for i, (_, real_inputs, real_targets) in enumerate(self.train_loader_for_protonet):
                            real_inputs, real_targets = real_inputs.to(self._device), real_targets.to(self._device)
                        real_embeddings = self._network.backbone(real_inputs, 0)[:, 0]
                    sampled_data.append(real_embeddings)
                    sampled_label.extend(real_targets.cpu().numpy().tolist())
                # 采样完毕<==========================
                sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
                sampled_label = torch.tensor(sampled_label).long().to(self._device)
                inputs = sampled_data
                targets = sampled_label
                sf_indexes = torch.randperm(inputs.size(0))  # 随机打乱数据
                inputs = inputs[sf_indexes]
                targets = targets[sf_indexes]

                for _iter in range(self._total_classes):
                    if _iter == self._total_classes -1:
                        inp = inputs[_iter * num_sampled_pcls:]
                        tgt = targets[_iter * num_sampled_pcls:]
                    else:
                        inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                        tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                    outputs = self._network.fc.forward_few_shot_calib(inp,few_shot_margin=self.few_shot_margin,label=tgt)
                    logits = outputs['logits']

                    loss = F.cross_entropy(logits[:, :self._total_classes], tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                scheduler.step()

                test_acc = self._compute_accuracy(self._network, self.test_loader, test=True)
                info = 'Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, losses / self._total_classes, test_acc)
                logging.info(info)
            # 将self._network.fc.few_shot_weight 合并进 self._network.fc.weight, 并删除 few_shot_weight
            self._network.fc.weight = nn.Parameter(torch.cat([self._network.fc.weight.data,self._network.fc.few_shot_weight.data],dim=0))
            self._network.fc.few_shot_weight = None

        else:  # base session
            total_epoch = self.args['tuned_epoch']
            for _, epoch in enumerate(range(total_epoch)):
                if debug and _ > 0:
                    break
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    if debug and i > 0:
                        break
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    # if self._cur_task == 0:
                    out_1 = self._network(inputs, label=targets, first_flow=1)
                    logits = out_1["logits"]
                    loss_1 = F.cross_entropy(logits, targets)

                    logits_m = self._network(inputs, label=targets, first_flow=2)["logits"]
                    loss_m = F.cross_entropy(logits_m, targets)
                    loss = loss_1 + loss_m

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits_m, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                test_cur_acc = self._compute_accuracy(self._network, self.test_curr_loader,first_flow=90)
                # test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {} => Loss {:.3f}, Train_accy {:.2f}, Test_curr_accy {:.2f},".format(
                    self._cur_task,
                    epoch + 1,
                    losses / len(train_loader),
                    train_acc,
                    test_cur_acc
                )
                logging.info(info)

    def _compute_accuracy(self, model, loader,test=False,first_flow=0):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs,test=test,first_flow=first_flow)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

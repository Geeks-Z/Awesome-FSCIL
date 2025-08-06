from typing import Iterable, Optional, List

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from CoLeLib.training.templates.base import BaseTemplate
from CoLeLib.evaluation import Evaluator
from CoLeLib.models import IncrementalClassifier


class SupervisedTemplate(BaseTemplate):
    def __init__(
        self,
        seed,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        gradient_accumulation_steps: int = 1,
        grad_clip_max_norm=None,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        evaluation_metrics: List[str] = ["acc", "loss", "forgetting"],
        loggers: List[str] = ["interactive", "json"],
        json_file_name: str = "results.json",
    ):
        super().__init__(model=model, device=device, seed=seed)

        self.optimizer = optimizer
        self._criterion = criterion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_max_norm = grad_clip_max_norm
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.eval_mb_size = eval_mb_size

        self.epoch = None

        self.dataloader = None

        self.mb = None
        self.mb_output = None

        self.loss = None

        if not isinstance(evaluation_metrics, Iterable):
            evaluation_metrics = [evaluation_metrics]

        if not isinstance(loggers, Iterable):
            loggers = [loggers]

        self.evaluator = Evaluator(evaluation_metrics, loggers, json_file_name)

        # 时间统计相关变量
        self.time_stats = None
        self.current_exp_train_start = None
        self.current_exp_eval_start = None

    @property
    def mb_x(self):
        return self.mb[0].to(self.device)

    @property
    def mb_y(self):
        return self.mb[1].long().to(self.device)

    def criterion(self):
        return self._criterion(self.mb_output, self.mb_y)

    def forward(self):
        return self.model(self.mb_x)

    def _before_training_exp(self):
        super()._before_training_exp()
        self.evaluator.update_metrics(self, "before_training_exp")
        self.evaluator.update_loggers(
            self, "before_training_exp", self.evaluator.metrics
        )
        self.is_training = True
        self.model.train()
        self.dataloader = DataLoader(
            self.experience,
            batch_size=self.train_mb_size,
            shuffle=True,
            pin_memory=True,
            num_workers=12,
        )
        self.model_adaptation()
        self.make_optimizer()

        # 记录当前experience训练开始时间
        self.current_exp_train_start = time.time()
        self.start = time.time()

    def _train_exp(self):
        self.evaluator.update_metrics(self, "train_exp")
        self.evaluator.update_loggers(self, "train_exp", self.evaluator.metrics)
        for self.epoch in range(self.train_epochs):
            self._before_training_epoch()
            self._training_epoch()
            self._after_training_epoch()
        super()._train_exp()

    def _before_eval_exp(self):
        super()._before_eval_exp()
        self.dataloader = DataLoader(
            self.experience,
            batch_size=self.eval_mb_size,
            shuffle=False,
            pin_memory=True,
            num_workers=12,
        )
        self.evaluator.update_metrics(self, "before_eval_exp")
        self.evaluator.update_loggers(self, "before_eval_exp", self.evaluator.metrics)

    def _eval_exp(self):
        super()._eval_exp()
        for self.mb in self.dataloader:
            self._before_eval_iteration()
            with torch.no_grad():
                self.mb_output = self.forward()
            self.loss = self.criterion()

            self._after_eval_iteration()

    def _before_training_epoch(self):
        self.evaluator.update_metrics(self, "before_training_epoch")
        self.evaluator.update_loggers(
            self, "before_training_epoch", self.evaluator.metrics
        )

    def _training_epoch(self):
        for idx, self.mb in enumerate(self.dataloader):

            self._before_training_iteration()

            self.loss = 0

            self._before_forward()
            self.mb_output = self.forward()
            self._after_forward()

            self.loss += self.criterion() / self.gradient_accumulation_steps

            self._before_backward()
            self.loss.backward()
            self._after_backward()

            if ((idx + 1) % self.gradient_accumulation_steps == 0) or (
                idx + 1 == len(self.dataloader)
            ):
                self._before_update()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_max_norm
                    )
                self.optimizer.step()
                self._after_update()
                self.optimizer.zero_grad()

            self._after_training_iteration()

    def _after_training_epoch(self):
        self.evaluator.update_metrics(self, "after_training_epoch")
        self.evaluator.update_loggers(
            self, "after_training_epoch", self.evaluator.metrics
        )

    def _before_training_iteration(self):
        pass

    def _before_forward(self):
        pass

    def _after_forward(self):
        pass

    def _before_backward(self):
        pass

    def _after_backward(self):
        pass

    def _before_update(self):
        pass

    def _after_update(self):
        pass

    def _after_training_iteration(self):
        self.evaluator.update_metrics(self, "after_training_iteration")
        self.evaluator.update_loggers(
            self, "after_training_iteration", self.evaluator.metrics
        )

    def _after_training_exp(self):
        # 计算当前experience的训练时间
        exp_train_time = time.time() - self.current_exp_train_start
        print(
            f"Experience {self.num_actual_experience} 训练时间: {exp_train_time:.2f} 秒"
        )

        # 更新时间统计
        if self.time_stats is not None:
            self.time_stats["training_time"] += exp_train_time
            # 如果experiences_stats列表长度不够，扩展它
            while (
                len(self.time_stats["experiences_stats"]) < self.num_actual_experience
            ):
                self.time_stats["experiences_stats"].append(
                    {
                        "experience_id": len(self.time_stats["experiences_stats"]) + 1,
                        "training_time": 0.0,
                        "inference_time": 0.0,
                    }
                )
            self.time_stats["experiences_stats"][self.num_actual_experience - 1][
                "training_time"
            ] = exp_train_time

        super()._after_training_exp()
        self.evaluator.update_metrics(self, "after_training_exp")
        self.evaluator.update_loggers(
            self, "after_training_exp", self.evaluator.metrics
        )

    def _before_eval(self):
        super()._before_eval()
        self.evaluator.update_metrics(self, "before_eval")
        self.evaluator.update_loggers(self, "before_eval", self.evaluator.metrics)
        # 记录整个评估阶段的开始时间
        self.current_exp_eval_start = time.time()
        self.eval_start = time.time()

    def _after_eval(self):
        # 计算总推理时间
        total_eval_time = time.time() - self.current_exp_eval_start
        print(f"总推理时间: {total_eval_time:.2f} 秒")

        # 更新时间统计
        if self.time_stats is not None:
            self.time_stats["inference_time"] += total_eval_time
            # 将推理时间分配给当前已训练的所有experiences
            # 这里假设推理时间应该分配给最后一个experience
            if len(self.time_stats["experiences_stats"]) > 0:
                last_exp_idx = len(self.time_stats["experiences_stats"]) - 1
                self.time_stats["experiences_stats"][last_exp_idx][
                    "inference_time"
                ] = total_eval_time

        super()._after_eval()
        self.evaluator.update_metrics(self, "after_eval")
        self.evaluator.update_loggers(self, "after_eval", self.evaluator.metrics)

    def _after_eval_exp(self):
        super()._after_eval_exp()
        self.evaluator.update_metrics(self, "after_eval_exp")
        self.evaluator.update_loggers(self, "after_eval_exp", self.evaluator.metrics)

    def _after_training(self):
        super()._after_training()
        self.evaluator.update_metrics(self, "after_training")
        self.evaluator.update_loggers(self, "after_training", self.evaluator.metrics)

    def _before_eval_iteration(self):
        pass

    def _after_eval_iteration(self):
        self.evaluator.update_metrics(self, "after_eval_iteration")
        self.evaluator.update_loggers(
            self, "after_eval_iteration", self.evaluator.metrics
        )

    def make_optimizer(self):
        self.optimizer.param_groups[0]["params"] = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )

    def model_adaptation(self, model=None):
        if model is None:
            model = self.model
        for module in model.modules():
            if isinstance(module, IncrementalClassifier):
                module.adaptation(
                    sum(self.num_classes_per_exp[: self.num_actual_experience])
                )

        return model.to(self.device)

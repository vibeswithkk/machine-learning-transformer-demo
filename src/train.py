import os
import json
import time
import logging
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from .model import TransformerClassifier, ModelConfig, SimpleTokenizer
from .utils import setup_logging


class TrainingConfig:
    def __init__(
        self,
        output_dir: str = "./results",
        overwrite_output_dir: bool = False,
        do_train: bool = True,
        do_eval: bool = True,
        do_predict: bool = False,
        evaluation_strategy: str = "steps",
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        num_train_epochs: int = 3,
        max_steps: int = -1,
        warmup_steps: int = 0,
        return_outputs: bool = False,
        eval_steps: int = 500,
        logging_dir: str = "./logs",
        logging_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: int = 2,
        seed: int = 42,
        fp16: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = True,
        disable_tqdm: bool = False,
        remove_unused_columns: bool = True,
        label_names: Optional[List[str]] = None,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        gradient_checkpointing: bool = False,
        include_inputs_for_metrics: bool = False,
        use_legacy_prediction_loop: bool = False,
        push_to_hub: bool = False,
        resume_from_checkpoint: Optional[str] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        hub_strategy: str = "every_save",
        gradient_clipping: float = 1.0,
        lr_scheduler_type: str = "linear",
        warmup_ratio: float = 0.0,
        logging_first_step: bool = False,
        logging_nan_inf_filter: bool = True,
        save_on_each_node: bool = False,
        no_cuda: bool = False,
        use_mps_device: bool = False,
        seed_for_dataloader: int = 0,
    ):
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.evaluation_strategy = evaluation_strategy
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.return_outputs = return_outputs
        self.eval_steps = eval_steps
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.seed = seed
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.disable_tqdm = disable_tqdm
        self.remove_unused_columns = remove_unused_columns
        self.label_names = label_names
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.gradient_checkpointing = gradient_checkpointing
        self.include_inputs_for_metrics = include_inputs_for_metrics
        self.use_legacy_prediction_loop = use_legacy_prediction_loop
        self.push_to_hub = push_to_hub
        self.resume_from_checkpoint = resume_from_checkpoint
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        self.hub_strategy = hub_strategy
        self.gradient_clipping = gradient_clipping
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.logging_first_step = logging_first_step
        self.logging_nan_inf_filter = logging_nan_inf_filter
        self.save_on_each_node = save_on_each_node
        self.no_cuda = no_cuda
        self.use_mps_device = use_mps_device
        self.seed_for_dataloader = seed_for_dataloader


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EarlyStoppingCallback:
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.early_stopping_counter = 0
        self.should_stop = False
        
    def check_metric_value(self, metrics: Dict[str, float], metric_key: str, greater_is_better: bool):
        metric_value = metrics.get(metric_key)
        if metric_value is None:
            return
            
        if self.best_metric is None:
            self.best_metric = metric_value
            return
            
        if greater_is_better:
            if metric_value > self.best_metric * (1 + self.early_stopping_threshold):
                self.best_metric = metric_value
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
        else:
            if metric_value < self.best_metric * (1 - self.early_stopping_threshold):
                self.best_metric = metric_value
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.should_stop = True


class Trainer:
    def __init__(
        self,
        model: TransformerClassifier,
        args: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        compute_metrics=None,
        fp16: bool = False,
        local_rank: int = -1,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.fp16 = fp16
        self.local_rank = local_rank
        
        self.logger = setup_logging(__name__)
        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        self.best_metric = None
        self.best_model_checkpoint = None
        
        self._setup_device()
        self._setup_optimizer()
        self._setup_lr_scheduler()
        
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.args.early_stopping_patience,
            early_stopping_threshold=self.args.early_stopping_threshold
        )
        
        self._setup_output_dir()
        
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        if self.local_rank != -1:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        
    def _setup_device(self):
        if self.args.no_cuda or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.n_gpu = 0
        elif self.args.use_mps_device and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.n_gpu = 1
        else:
            if self.local_rank == -1:
                self.device = torch.device("cuda")
                self.n_gpu = torch.cuda.device_count()
            else:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.n_gpu = 1
                torch.distributed.init_process_group(backend='nccl')
                
        self.model.to(self.device)
        
    def _setup_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        
    def _setup_lr_scheduler(self):
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
        else:
            t_total = (
                len(self.train_dataset)
                // self.args.per_device_train_batch_size
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            
        if self.args.warmup_steps > 0:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = int(t_total * self.args.warmup_ratio)
            
        if self.args.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(t_total - current_step) / float(max(1, t_total - warmup_steps))
                )
                
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
            
    def _setup_output_dir(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(self.args.logging_dir, exist_ok=True)
        
    def _set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)
            
    def get_train_dataloader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.args.seed_for_dataloader)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=generator,
        )
        
    def get_eval_dataloader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.args.seed_for_dataloader)
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=generator,
        )
        
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_classes), labels.view(-1))
        
        return (loss, outputs) if self.args.return_outputs else (loss, None)
        
    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        if self.fp16 and self.scaler is not None:
            with autocast():
                loss, outputs = self.compute_loss(model, inputs)
        else:
            loss, outputs = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        if self.fp16 and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.detach()
        
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs
        
    def is_world_process_zero(self) -> bool:
        return True
        
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.args.logging_steps > 0 and (self.global_step % self.args.logging_steps == 0):
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            logs["loss"] = round(tr_loss_scalar / self.args.logging_steps, 4)
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            logs["epoch"] = round(epoch, 2)
            
            self.logger.info(json.dumps(logs))
            tr_loss -= tr_loss
            
        if self.args.save_steps > 0 and (self.global_step % self.args.save_steps == 0):
            self.save_model()
            
        if self.args.evaluation_strategy == "steps" and (self.global_step % self.args.eval_steps == 0):
            metrics = self.evaluate()
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            if self.args.load_best_model_at_end:
                metric_key = self.args.metric_for_best_model
                greater_is_better = self.args.greater_is_better
                
                metric_value = metrics.get(metric_key)
                if metric_value is not None:
                    if self.best_metric is None:
                        self.best_metric = metric_value
                        self.save_model(is_best=True)
                    else:
                        if greater_is_better:
                            if metric_value > self.best_metric:
                                self.best_metric = metric_value
                                self.save_model(is_best=True)
                        else:
                            if metric_value < self.best_metric:
                                self.best_metric = metric_value
                                self.save_model(is_best=True)
                
                self.early_stopping_callback.check_metric_value(metrics, metric_key, greater_is_better)
                
                if self.early_stopping_callback.should_stop:
                    self.logger.info("Early stopping triggered")
                    return True
                    
        return False
        
    def save_model(self, is_best=False):
        if is_best:
            output_dir = os.path.join(self.args.output_dir, "best_model")
        else:
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.__dict__ if hasattr(self.model.config, "__dict__") else self.model.config,
            "classes": list(self.tokenizer.vocab.keys()) if self.tokenizer else [],
            "tokenizer_path": "built-in",
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }, os.path.join(output_dir, "model.pt"))
        
        self.logger.info(f"Model saved to {output_dir}")
        
        if is_best:
            self.best_model_checkpoint = output_dir
        
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        
        total_loss = 0.0
        total_preds = []
        total_labels = []
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                labels = inputs.pop("labels")
                inputs = self._prepare_inputs(inputs)
                
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model.config.num_classes), labels.view(-1))
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(total_labels, total_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            total_labels, total_preds, average="weighted"
        )
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "epoch": self.epoch,
        }
        
        return metrics
        
    def train(self):
        self._set_seed()
        
        train_dataloader = self.get_train_dataloader()
        
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            num_train_epochs = self.args.num_train_epochs
            
        self.logger.info(f"***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size = {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {t_total}")
        
        tr_loss = torch.tensor(0.0)
        self.model.zero_grad()
        
        epoch_iterator = tqdm(range(num_train_epochs), desc="Epochs")
        
        for epoch in epoch_iterator:
            self.epoch = epoch
            train_dataloader_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=False)
            
            for step, inputs in enumerate(train_dataloader_iter):
                tr_loss_step = self.training_step(self.model, inputs)
                tr_loss += tr_loss_step
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
                ):
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.fp16 and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm
                            )
                            self.optimizer.step()
                    else:
                        if self.fp16 and self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                    should_stop = self._maybe_log_save_evaluate(tr_loss, self.model, None, epoch)
                    if should_stop:
                        break
                        
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
                    
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
                
        self.logger.info(f"\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return self.global_step, tr_loss.item() / self.global_step
        

def create_dummy_dataset(num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    texts = [
        "This is a positive example with good sentiment",
        "Negative sentiment expressed in this text",
        "Neutral statement without strong emotion",
        "Another positive sentiment example",
        "Clearly negative feedback provided here",
    ] * (num_samples // 5 + 1)
    
    labels = [0, 1, 2, 0, 1] * (num_samples // 5 + 1)
    
    return texts[:num_samples], labels[:num_samples]
    

def train_model(
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    train_texts: Optional[List[str]] = None,
    train_labels: Optional[List[int]] = None,
    eval_texts: Optional[List[str]] = None,
    eval_labels: Optional[List[int]] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> Tuple[TransformerClassifier, Trainer]:
    
    if model_config is None:
        model_config = ModelConfig()
        
    if training_config is None:
        training_config = TrainingConfig()
        
    model = TransformerClassifier(model_config)
    
    if tokenizer is not None:
        model.tokenizer = tokenizer
        
    tokenizer_to_use = model.tokenizer
    
    if train_texts is None or train_labels is None:
        train_texts, train_labels = create_dummy_dataset(100)
        
    if eval_texts is None or eval_labels is None:
        eval_texts, eval_labels = create_dummy_dataset(20)
        
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer_to_use)
    eval_dataset = TextClassificationDataset(eval_texts, eval_labels, tokenizer_to_use)
    
    trainer = Trainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer_to_use,
    )
    
    trainer.train()
    
    return model, trainer
# 更新runner：适配deepspeed

import datetime
import math
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch_npu
import torch.distributed as dist
import webdataset as wds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset, IterableDataset

from q_former.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from base_dataset import ConcatDataset as LAVISConcatDataset
try:
    from dataset import custom_collate_fn as _NAV_COLLATE_FN
except Exception:
    _NAV_COLLATE_FN = None

try:
    import deepspeed
except Exception:
    deepspeed = None

### 如果有更多dataset 对应的各种其它场景下的任务
### 请在此处补充dataset import

# ======== Learning rate schedulers  ======== #
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# ======== Learning rate schedulers  ======== #
class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


LR_SCHEDULERS = {
    "linear_warmup_step_lr": LinearWarmupStepLRScheduler,
    "linear_warmup_cosine_lr": LinearWarmupCosineLRScheduler,
}

def move_to_device(sample: Any, device: torch.device):
    def _move(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_move(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_move(v) for v in x)
        return x

    return _move(sample)


def record_npu_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.npu.current_stream())
    elif isinstance(batch, (list, tuple)):
        for t in batch:
            record_npu_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_npu_stream(t)


class MultiIterLoader:
    # VLN训练非多数据集训练，未启用
    # 如若之后有别的任务需要使用多数据集训练，请使用该Loader并启用相对应配置
    def __init__(self, loaders: List[Iterable], ratios: Optional[List[float]] = None):
        for loader in loaders:
            assert hasattr(loader, "__next__"), f"Loader {loader} has no __next__ method."

        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(r) / sum(ratios) for r in ratios]

        self.loaders = loaders
        self.ratios = ratios

    def __next__(self):
        loader_idx = torch.multinomial(torch.tensor(self.ratios), num_samples=1).item()
        return next(self.loaders[loader_idx])

    def __iter__(self):
        return self


class PrefetchLoader:

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.npu.Stream() if device.type == "npu" else None

    def __iter__(self):
        if self.device.type != "npu":
            for batch in self.loader:
                yield move_to_device(batch, self.device)
            return

        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return

        with torch.npu.stream(self.stream):
            self.batch = move_to_device(self.batch, self.device)

    def next(self, it):
        torch.npu.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_npu_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        return getattr(self.loader, name)


class IterLoader:

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def reorg_datasets_by_split(datasets: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
    reorg = dict()
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            reorg.setdefault(split_name, []).append(dataset_split)
    return reorg


def concat_datasets(datasets: Dict[str, List[Any]]):
    for split_name in datasets:
        if split_name != "train":
            assert len(datasets[split_name]) == 1, f"Do not support multiple {split_name} datasets."
            datasets[split_name] = datasets[split_name][0]
            continue

        iterable_datasets, map_datasets = [], []
        for dataset in datasets[split_name]:
            if isinstance(dataset, wds.DataPipeline):
                iterable_datasets.append(dataset)
            elif isinstance(dataset, IterableDataset):
                raise NotImplementedError("Do not support concatenation of generic IterableDataset.")
            else:
                map_datasets.append(dataset)

        chained = ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
        concat = LAVISConcatDataset(map_datasets) if len(map_datasets) > 0 else None

        train_datasets: Tuple[Any, ...] = tuple(x for x in (concat, chained) if x is not None)
        train_datasets = train_datasets[0] if len(train_datasets) == 1 else train_datasets
        datasets[split_name] = train_datasets

    return datasets

# ======== Config helpers ======== #
def load_yaml_config(cfg_path: str, options: Optional[List[str]] = None):
    cfg = OmegaConf.load(cfg_path)
    if options:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(options))
    return cfg

### 如果有其它任务/数据需要训练，请按照此方式定义runner 
### TODO：如果要定义其它runner，建议维护一个注册表

class PGNRunner:
    def __init__(self, cfg, task, model, datasets, job_id: Optional[str] = None, base_output_dir: Optional[str] = None):
        self.config = cfg
        self.job_id = job_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.task = task
        self.datasets = datasets

        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._ds_engine = None # deepspeed engine
        self._ds_config = None # deepspeed config
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path.cwd()
        self.setup_output_dir()

    # -------- Config accessors -------- #
    @property
    def run_cfg(self):
        return getattr(self.config, "run", None) or self.config.get("run")

    def _get_run_attr(self, key, default=None):
        run = self.run_cfg
        if run is None:
            return default
        if isinstance(run, dict):
            return run.get(key, default)
        return getattr(run, key, default)

    # -------- Core properties -------- #
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self._get_run_attr("device", "npu"))
        return self._device

    @property
    def use_distributed(self):
        return bool(self._get_run_attr("distributed", False))

    @property
    def use_deepspeed(self):
        return bool(self._get_run_attr("use_deepspeed", False))

    @property
    def model(self):
        model_device = getattr(self._model, "device", None)
        if model_device is None:
            try:
                model_device = next(self._model.parameters()).device
            except StopIteration:
                model_device = self.device

        if model_device != self.device and not self.use_deepspeed:
            self._model = self._model.to(self.device)

        if self._wrapped_model is None:
            if self.use_deepspeed:
                self._wrapped_model = self._init_deepspeed_engine()
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.use_deepspeed:
                if self._ds_engine is None:
                    return None
                self._optimizer = self._ds_engine.optimizer
            else:
                self._optimizer = torch.optim.AdamW(
                    params=self.model.parameters(),
                    lr=float(self._get_run_attr("init_lr")),
                    weight_decay=float(self._get_run_attr("weight_decay", 0.0)),
                )
        return self._optimizer

    @property
    def scaler(self):
        if self.use_deepspeed:
            return None
        amp = self._get_run_attr("amp", False)
        if amp and self._scaler is None:
            self._scaler = torch.npu.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            sched_name = self._get_run_attr("lr_sched")
            if sched_name not in LR_SCHEDULERS:
                raise ValueError(f"Unsupported lr_sched '{sched_name}'. Available: {list(LR_SCHEDULERS.keys())}")
            lr_sched_cls = LR_SCHEDULERS[sched_name]
            optimizer = self.optimizer
            if optimizer is None:
                raise RuntimeError("DeepSpeed optimizer is not initialized; access model before lr_scheduler.")

            self._lr_sched = lr_sched_cls(
                optimizer=optimizer,
                max_epoch=self.max_epoch,
                min_lr=self.min_lr,
                init_lr=self.init_lr,
                decay_rate=self._get_run_attr("lr_decay_rate", 1),
                warmup_start_lr=self._get_run_attr("warmup_lr", -1),
                warmup_steps=self._get_run_attr("warmup_steps", 0),
            )
        return self._lr_sched

    @property
    def dataloaders(self) -> Dict[str, Any]:
        if self._dataloaders is None:
            datasets = self._normalize_datasets()
            if not datasets:
                raise ValueError("No datasets found for runner.")

            num_workers = int(self._get_run_attr("num_workers", 4))
            batch_size_train = int(
                self._get_run_attr(
                    "batch_size_train",
                    self._get_run_attr("batch_size", 1),
                )
            )
            batch_size_eval = int(
                self._get_run_attr(
                    "batch_size_eval",
                    self._get_run_attr("batch_size", batch_size_train),
                )
            )

            train_splits = self.train_splits or (["train"] if "train" in datasets else [])
            loaders = {}
            for split_name, dataset in datasets.items():
                is_train = split_name in train_splits or (not train_splits and split_name == "train")
                batch_size = batch_size_train if is_train else batch_size_eval
                if isinstance(dataset, (list, tuple)):
                    collate_fn = [self._resolve_collate_fn(d) for d in dataset]
                else:
                    collate_fn = self._resolve_collate_fn(dataset)

                dataset_ratios = self._resolve_dataset_ratios(split_name, is_train)
                loader = self.create_loaders(
                    datasets=[dataset],
                    num_workers=num_workers,
                    batch_sizes=[batch_size],
                    is_trains=[is_train],
                    collate_fns=[collate_fn],
                    dataset_ratios=dataset_ratios,
                )[0]
                loaders[split_name] = loader

            self._dataloaders = loaders
        return self._dataloaders

    @property
    def npu_enabled(self):
        return self.device.type == "npu"

    @property
    def max_epoch(self):
        return int(self._get_run_attr("max_epoch"))

    @property
    def log_freq(self):
        return int(self._get_run_attr("log_freq", 50))

    @property
    def init_lr(self):
        return float(self._get_run_attr("init_lr"))

    @property
    def min_lr(self):
        return float(self._get_run_attr("min_lr"))

    @property
    def accum_grad_iters(self):
        return int(self._get_run_attr("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        return self._get_run_attr("valid_splits", []) or []

    @property
    def test_splits(self):
        return self._get_run_attr("test_splits", []) or []

    @property
    def train_splits(self):
        return self._get_run_attr("train_splits", []) or []

    @property
    def evaluate_only(self):
        return bool(self._get_run_attr("evaluate", False))

    @property
    def use_dist_eval_sampler(self):
        return bool(self._get_run_attr("use_dist_eval_sampler", True))

    @property
    def resume_ckpt_path(self):
        return self._get_run_attr("resume_ckpt_path", None)

    @property
    def train_loader(self):
        return self.dataloaders["train"]
    
    def setup_output_dir(self):
        output_dir = self.base_output_dir / self._get_run_attr("output_dir") / self.job_id
        result_dir = output_dir / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = result_dir
        self.output_dir = output_dir

    # -------- Core loop -------- #
    def train(self):
        start_time = time.time()
        best_agg_metric = float("-inf")
        best_epoch = 0

        self.log_config()

        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info(f"Evaluating on {split_name}.")
                    val_log = self.eval_epoch(split_name=split_name, cur_epoch=cur_epoch)
                    if val_log is not None and is_main_process():
                        assert "agg_metrics" in val_log, "No agg_metrics found in validation log."
                        agg_metrics = val_log["agg_metrics"]
                        if agg_metrics > best_agg_metric and split_name == "val":
                            best_epoch, best_agg_metric = cur_epoch, agg_metrics
                            self._save_checkpoint(cur_epoch, is_best=True)
                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)
            else:
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False) # 没有验证集，非仅验证模式，选择当前epoch直接save一次

            if self.evaluate_only:
                break

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Training time {total_time_str}")

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()
        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                if self.dataloaders.get(split_name) is None:
                    logging.warning(f"Skip test split '{split_name}': no dataset/dataloader configured.")
                    continue
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )
            return test_logs

    def train_epoch(self, epoch):
        self.model.train()
        if self._has_task_method("train_epoch"):
            return self.task.train_epoch(
                epoch=epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                npu_enabled=self.npu_enabled,
                log_freq=self.log_freq,
                accum_grad_iters=self.accum_grad_iters,
            )
        return self._default_train_epoch(epoch)

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, f"data_loader for split {split_name} is None."

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        if not self._has_task_method("evaluation"):
            return self._default_eval_epoch(split_name=split_name, cur_epoch=cur_epoch)

        if self._has_task_method("before_evaluation"):
            self.task.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            if self._has_task_method("after_evaluation"):
                return self.task.after_evaluation(
                    val_result=results,
                    split_name=split_name,
                    epoch=cur_epoch,
                )
            return results

    def unwrap_dist_model(self, model):
        if self.use_deepspeed:
            return model
        return model

    def _prepare_deepspeed_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ds_cfg = dict(config)
        micro_bsz = int(self._get_run_attr("batch_size_train", self._get_run_attr("batch_size", 1)))
        accum_steps = int(self.accum_grad_iters)
        ds_cfg["train_micro_batch_size_per_gpu"] = micro_bsz
        ds_cfg["gradient_accumulation_steps"] = accum_steps

        world = get_world_size() if self.use_distributed else 1
        ds_cfg.setdefault("train_batch_size", micro_bsz * accum_steps * world)

        opt_cfg = ds_cfg.get("optimizer")
        if opt_cfg is None:
            opt_cfg = {"type": "AdamW", "params": {}}
            ds_cfg["optimizer"] = opt_cfg
        opt_params = opt_cfg.setdefault("params", {})
        opt_params.setdefault("betas", [0.9, 0.999])
        opt_params.setdefault("eps", 1e-8)
        opt_params["lr"] = float(self._get_run_attr("init_lr"))
        opt_params["weight_decay"] = float(self._get_run_attr("weight_decay", 0.0))

        return ds_cfg

    def _load_deepspeed_config(self) -> Dict[str, Any]:
        if self._ds_config is not None:
            return self._ds_config
        ds_cfg = self._get_run_attr("deepspeed_config", None)
        if ds_cfg is None:
            raise ValueError("run.deepspeed_config is required when use_deepspeed=True.")
        if isinstance(ds_cfg, str):
            path = Path(ds_cfg)
            if not path.is_absolute():
                path = Path.cwd() / ds_cfg
            with open(path, "r") as f:
                config = json.load(f)
        elif isinstance(ds_cfg, dict):
            config = dict(ds_cfg)
        else:
            raise ValueError("run.deepspeed_config must be a path or a dict.")

        self._ds_config = self._prepare_deepspeed_config(config)
        return self._ds_config

    def _init_deepspeed_engine(self):
        if self._ds_engine is not None:
            return self._ds_engine
        if deepspeed is None:
            raise RuntimeError("DeepSpeed is not installed. Install deepspeed==0.18.3.")

        ds_cfg = self._load_deepspeed_config()
        dist_init_required = not (dist.is_available() and dist.is_initialized())
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self._model,
            model_parameters=trainable_params,
            config=ds_cfg,
            dist_init_required=dist_init_required,
        )
        self._ds_engine = engine
        self._optimizer = optimizer
        return engine

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            if isinstance(dataset, (ChainDataset, wds.DataPipeline)):
                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)
            else:
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader, device=self.device)
                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)
            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, (list, tuple)):
                loader = MultiIterLoader(
                    loaders=[_create_loader(d, num_workers, bsz, is_train, collate_fn[i]) for i, d in enumerate(dataset)],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)
        return loaders

    # -------- Checkpointing -------- #
    def _save_checkpoint(self, cur_epoch, is_best=False):
        if not self.use_deepspeed and not is_main_process():
            return
        if self.use_deepspeed:
            tag = f"checkpoint_{'best' if is_best else cur_epoch}"
            client_state = {
                "config": self._config_to_dict(self.config),
                "epoch": cur_epoch,
            }
            logging.info("Saving DeepSpeed checkpoint at epoch %s to %s (tag=%s).", cur_epoch, self.output_dir, tag)
            self.model.save_checkpoint(str(self.output_dir), tag=tag, client_state=client_state)
            return
        save_obj = {
            "model": self.unwrap_dist_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self._config_to_dict(self.config),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            f"checkpoint_{'best' if is_best else cur_epoch}.pth",
        )
        logging.info(f"Saving checkpoint at epoch {cur_epoch} to {save_to}.")
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        if self.use_deepspeed:
            tag = "checkpoint_best"
            logging.info("Loading DeepSpeed checkpoint from %s (tag=%s).", self.output_dir, tag)
            self.model.load_checkpoint(str(self.output_dir), tag=tag)
            return self.model
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info(f"Loading checkpoint from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        return model

    def _load_checkpoint(self, url_or_filename):
        if self.use_deepspeed:
            if not os.path.isdir(url_or_filename):
                raise RuntimeError("resume_ckpt_path must be a directory when using DeepSpeed.")
            logging.info("Loading DeepSpeed checkpoint from %s.", url_or_filename)
            _, client_state = self.model.load_checkpoint(url_or_filename)
            if client_state and "epoch" in client_state:
                self.start_epoch = client_state["epoch"] + 1
            return
        if isinstance(url_or_filename, str) and url_or_filename.startswith(("http://", "https://")):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Resume checkpoint from {url_or_filename}")

    # -------- Logging -------- #
    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self._config_to_dict(self.config), indent=4) + "\n")

    def _config_to_dict(self, cfg):
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "to_container"):
            return cfg.to_container()
        try:
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return dict(cfg)

    def _normalize_datasets(self) -> Dict[str, Any]:
        if self.datasets is None:
            return {}
        if isinstance(self.datasets, dict):
            if all(not isinstance(v, dict) for v in self.datasets.values()):
                return self.datasets
            return concat_datasets(reorg_datasets_by_split(self.datasets))
        return {"train": self.datasets}

    def _resolve_collate_fn(self, dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "collater"):
            return dataset.collater
        if hasattr(dataset, "collate_fn"):
            return dataset.collate_fn
        if _NAV_COLLATE_FN is not None and dataset.__class__.__name__ == "NavigationDataset":
            return _NAV_COLLATE_FN
        return None

    def _resolve_dataset_ratios(self, split_name: str, is_train: bool):
        dataset_ratios = self._get_run_attr("dataset_ratios", None)
        if dataset_ratios is None:
            dataset_ratios = self._get_run_attr("train_dataset_ratios", None)
        if isinstance(dataset_ratios, dict):
            return dataset_ratios.get(split_name)
        return dataset_ratios if is_train else None

    def _has_task_method(self, method_name: str) -> bool:
        return self.task is not None and hasattr(self.task, method_name)

    def _default_train_epoch(self, epoch: int) -> Dict[str, float]:
        data_loader = self.train_loader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        accum_grad_iters = max(self.accum_grad_iters, 1)
        amp = bool(self._get_run_attr("amp", False))
        use_deepspeed = self.use_deepspeed

        if amp and not use_deepspeed:
            if self.npu_enabled:
                autocast_ctx = torch.npu.amp.autocast()
            elif self.device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast()
            else:
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = nullcontext()

        if use_deepspeed:
            model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_steps = 0

        if isinstance(data_loader, IterLoader):
            data_iter = (next(data_loader) for _ in range(len(data_loader)))
        else:
            data_iter = iter(data_loader)

        total_iters = len(data_loader) if hasattr(data_loader, "__len__") else None
        for step, samples in enumerate(data_iter):
            with autocast_ctx:
                outputs = model(samples)
                if isinstance(outputs, dict):
                    loss = outputs.get("loss")
                elif isinstance(outputs, (list, tuple)):
                    loss = outputs[0]
                else:
                    loss = outputs
                if loss is None:
                    raise RuntimeError("Model output has no loss for training.")
            loss_value = float(loss.detach().item())
            total_loss += loss_value
            total_steps += 1

            if use_deepspeed:
                model.backward(loss)
                model.step()
                if self.lr_scheduler is not None and model.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step(cur_epoch=epoch, cur_step=step)
            else:
                loss = loss / accum_grad_iters
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum_grad_iters == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(cur_epoch=epoch, cur_step=step)

            if self.log_freq and (step + 1) % self.log_freq == 0 and is_main_process():
                if total_iters is None:
                    logging.info(f"Epoch {epoch} step {step + 1} loss {loss_value:.4f}")
                else:
                    logging.info(
                        f"Epoch {epoch} step {step + 1}/{total_iters} loss {loss_value:.4f}"
                    )

        avg_loss = total_loss / max(1, total_steps)
        return {"loss": avg_loss}

    @torch.no_grad()
    def _default_eval_epoch(self, split_name: str, cur_epoch: int) -> Dict[str, float]:
        data_loader = self.dataloaders.get(split_name)
        if data_loader is None:
            raise RuntimeError(f"data_loader for split {split_name} is None.")

        model = self.unwrap_dist_model(self.model)
        model.eval()

        total_loss = 0.0
        total_steps = 0

        if isinstance(data_loader, IterLoader):
            data_iter = (next(data_loader) for _ in range(len(data_loader)))
        else:
            data_iter = iter(data_loader)

        for samples in data_iter:
            outputs = model(samples)
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
            elif isinstance(outputs, (list, tuple)):
                loss = outputs[0]
            else:
                loss = outputs
            if loss is None:
                raise RuntimeError("Model output has no loss for evaluation.")
            total_loss += float(loss.detach().item())
            total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        return {"agg_metrics": -avg_loss, "loss": avg_loss, "epoch": cur_epoch}

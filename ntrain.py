import argparse
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch_npu
from omegaconf import OmegaConf

# torch.autograd.set_detect_anomaly(True)

from q_former.dist_utils import download_cached_file, get_rank, init_distributed_mode
from q_former.logger import setup_logger
from q_former.pgnconfig import PGNMultiModalConfig
from q_former.registry import registry
from q_former.utils import is_url, now
from runner import PGNRunner, load_yaml_config

from dataset import NavigationDataset
from pgn import Blip2PanGu


def log_device(run_cfg):
    print(
        f"[dist] rank={get_rank()}, world={getattr(run_cfg, 'world_size', 1)}, "
        f"distributed={getattr(run_cfg, 'distributed', False)}, "
        f"dist_url={getattr(run_cfg, 'dist_url', 'n/a')}"
    )
    print(f"[env] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    print(
        f"[torch] torch={torch.__version__} "
        f"torch_npu={getattr(torch_npu, '__version__', 'not installed')}"
    )
    if getattr(run_cfg, "device", None) == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        idx = torch.npu.current_device()
        props = torch.npu.get_device_properties(idx)
        print(f"[device] NPU index={idx} name={torch.npu.get_device_name(idx)} total_mem={props.total_memory/1024**3:.1f}GB count={torch.npu.device_count()}")
    elif torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"[device] CUDA index={idx} name={torch.cuda.get_device_name(idx)} total_mem={torch.cuda.get_device_properties(idx).total_memory/1024**3:.1f}GB count={torch.cuda.device_count()}")
    else:
        print("[device] CPU mode")

def log_model(model, model_cfg: Optional[Dict[str, Any]] = None):
    arch = None
    if model_cfg:
        arch = model_cfg.get("arch")
    print(f"[model] top={model.__class__.__name__} from cfg model.arch={arch}")
    llm = getattr(model, "llm", None) or getattr(model, "model", None)
    if llm is not None:
        p = next(llm.parameters())
        total = sum(p.numel() for p in llm.parameters()) / 1e9
        trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad) / 1e9
        print(f"[model] llm_class={llm.__class__.__name__} dtype={p.dtype} device={p.device} params={total:.2f}B trainable={trainable:.2f}B")
        print(f"[model] llm_config_class={llm.config.__class__.__name__}")
        print(llm.config.to_json_string())  # Hugging Face 配置完整打印
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        print(f"[tokenizer] vocab={tok.vocab_size} pad={tok.pad_token_id} eos={tok.eos_token_id} file={getattr(tok, 'vocab_file', 'n/a')}")



def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    args = parser.parse_args()

    return args


def _resolve_dtype(dtype_value: Any) -> Optional[torch.dtype]:
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if not isinstance(dtype_value, str):
        return None
    name = dtype_value.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    return None


def _ensure_run_defaults(run_cfg: Any) -> Any:
    defaults = {
        "seed": 42,
        "dist_url": "env://",
        "distributed": True,
        "world_size": 1,
        "npu": 0,
        "device": "npu",
    }
    for key, value in defaults.items():
        if run_cfg.get(key, None) is None:
            run_cfg[key] = value
    return run_cfg


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    run_cfg = cfg.get("run", {})
    runner_name = run_cfg.get("runner", "runner_base")
    runner_cls = registry.get_runner_class(runner_name)
    return runner_cls or PGNRunner

def setup_seeds(run_cfg):
    seed = run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["ASCEND_DETERMINISTIC"] = "1"
    os.environ["TUNE_OPS_MODE"] = "0"


def build_datasets(cfg) -> Dict[str, Any]:
    dataset_cfg = cfg.get("datasets", {})
    if not dataset_cfg:
        raise ValueError("No datasets defined in config.")

    datasets = {}
    for split_name, split_cfg in dataset_cfg.items():
        if split_cfg is None:
            continue
        json_dir = split_cfg.get("json_dir")
        image_root_dir = split_cfg.get("image_root_dir")
        if not json_dir or not image_root_dir:
            raise ValueError(f"Dataset split '{split_name}' is missing json_dir or image_root_dir.")

        transform_cfg = split_cfg.get("transform", None)
        history_frames = split_cfg.get("history_frames", None)
        datasets[split_name] = NavigationDataset(
            json_dir=json_dir,
            image_root_dir=image_root_dir,
            transform=transform_cfg,
            history_frames=history_frames,
        )
    return datasets


def build_model(cfg, device: Optional[torch.device]):
    model_cfg = cfg.get("model", {}) or {}
    mm_config = PGNMultiModalConfig()
    if model_cfg:
        for key, value in model_cfg.items():
            if key == "dtype":
                resolved = _resolve_dtype(value)
                if resolved is not None:
                    mm_config.dtype = resolved
                continue
            if hasattr(mm_config, key):
                setattr(mm_config, key, value)
    pretrained_ckpt_path = model_cfg.get("pretrained_ckpt_path")
    if pretrained_ckpt_path: 
        # 训练配置中如有预训练权重，则覆写mm_config，保证有权重可以运行
        # 训练配置优先级高于多模态模型配置优先级
        mm_config.load_llm_pretrained = False
        mm_config.load_vision_pretrained = False
        mm_config.load_qformer_pretrained = False

    defaults = Blip2PanGu.__init__.__defaults__ or ()
    prompt0_default = defaults[0] if len(defaults) > 0 else None
    prompt2_default = defaults[1] if len(defaults) > 1 else None
    prompt1_default = defaults[2] if len(defaults) > 2 else None
    prompt_action_default = defaults[3] if len(defaults) > 3 else None
    prompt_answer_default = defaults[4] if len(defaults) > 4 else None
    model = Blip2PanGu(
        prompt0=model_cfg.get("prompt0", prompt0_default),
        prompt2=model_cfg.get("prompt2", prompt2_default),
        prompt1=model_cfg.get("prompt1", prompt1_default),
        prompt_action = model_cfg.get("prompt_action", prompt_action_default),
        prompt_answer = model_cfg.get("prompt_answer", prompt_answer_default),
        max_txt_len=model_cfg.get("max_txt_len", mm_config.max_txt_len),
        llm_hidden_size=model_cfg.get("llm_hidden_size", 4096),
        device=device,
        mm_config=mm_config,
        dtype=mm_config.dtype,
    )
    return model


def _pretty_print_cfg(cfg):
    try:
        print(OmegaConf.to_yaml(cfg))
    except Exception:
        print(cfg)

def _expand_token_weights(weight: torch.Tensor, target_vocab: int) -> torch.Tensor:
    old_vocab = weight.shape[0]
    if old_vocab == target_vocab:
        return weight
    if old_vocab > target_vocab:
        logging.warning(
            "Checkpoint vocab (%d) is larger than model vocab (%d); truncating.",
            old_vocab,
            target_vocab,
        )
        return weight[:target_vocab]
    new_weight = weight.new_empty((target_vocab, weight.shape[1]))
    new_weight[:old_vocab] = weight
    mean_vec = weight.mean(dim=0, keepdim=True)
    new_weight[old_vocab:] = mean_vec
    return new_weight

def _load_pretrained_weights(model, ckpt_path: str, map_location="cpu", strict: bool = True):
    if is_url(ckpt_path):
        cached_file = download_cached_file(ckpt_path, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location=map_location)
    elif os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=map_location)
    else:
        raise RuntimeError(f"checkpoint url or path is invalid: {ckpt_path}")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    for key in ("llm.model.embed_tokens.weight", "llm.lm_head.weight"):
        if key not in state_dict or key not in model_state:
            continue
        if state_dict[key].shape != model_state[key].shape:
            state_dict[key] = _expand_token_weights(state_dict[key], model_state[key].shape[0])

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and incompatible is not None:
        if incompatible.missing_keys:
            logging.info("Missing keys when loading pretrained weights: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logging.info("Unexpected keys when loading pretrained weights: %s", incompatible.unexpected_keys)
    logging.info("Loaded pretrained weights from %s", ckpt_path)

def main():

    job_id = now()

    args = parse_args()
    options = args.cfg_options or args.options
    cfg = load_yaml_config(args.cfg_path, options)

    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError("Config missing required 'run' section.")
    run_cfg = _ensure_run_defaults(run_cfg)

    init_distributed_mode(run_cfg)
    if run_cfg.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(run_cfg.npu)

    setup_seeds(run_cfg)
    setup_logger()
    log_device(run_cfg)
    _pretty_print_cfg(cfg)

    device = torch.device(run_cfg.device)
    use_deepspeed = bool(run_cfg.get("use_deepspeed", False)) if isinstance(run_cfg, dict) else bool(getattr(run_cfg, "use_deepspeed", False))
    build_device = None if use_deepspeed else device
    model = build_model(cfg, device=build_device)
    model_cfg = cfg.get("model", {}) or {}
    pretrained_ckpt_path = model_cfg.get("pretrained_ckpt_path")
    resume_ckpt_path = (
        run_cfg.get("resume_ckpt_path") if isinstance(run_cfg, dict) else getattr(run_cfg, "resume_ckpt_path", None)
    )
    if pretrained_ckpt_path:
        if resume_ckpt_path:
            logging.warning(
                "Both model.pretrained_ckpt_path and run.resume_ckpt_path are set; skipping pretrained load."
            )
        else:
            strict = model_cfg.get("pretrained_strict", True)
            _load_pretrained_weights(model, pretrained_ckpt_path, map_location="cpu", strict=bool(strict))
    log_model(model, cfg.get("model", {}))

    datasets = build_datasets(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        task=None,  # VLN项目的task默认为None，在没有设置其它task时，默认执行VLN任务逻辑，如果有其它task，请注册并实现
        model=model,
        datasets=datasets,
        job_id=job_id,
    )
    runner.train()


if __name__ == "__main__":
    main()

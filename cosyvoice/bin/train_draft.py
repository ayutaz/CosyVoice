# Copyright (c) 2025 Speech Speculative Decoding for CosyVoice3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a draft model for Speech Speculative Decoding.

Freeze strategy (paper-aligned):
  - layers[0], layers[1] (from target layers 0, 1): trainable
  - layers[2]~[7] (from target layers 18~23): frozen
  - llm_decoder: trainable
  - speech_embedding, embed_tokens: frozen

Loss: Cross-entropy only (Eq. 7 from the paper).

Usage:
    python cosyvoice/bin/train_draft.py \
        --train_engine torch_ddp \
        --model llm \
        --config examples/libritts/cosyvoice3/conf/cosyvoice3_draft.yaml \
        --train_data /path/to/train_data \
        --cv_data /path/to/cv_data \
        --model_dir ./checkpoints_draft \
        --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm_draft.pt
"""

from __future__ import print_function
import argparse
import datetime
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter,
    save_model,
    wrap_cuda_model,
    check_modify_and_save_config,
)


def freeze_draft_model(model):
    """Apply freeze strategy for draft model training.

    Trainable: layers[0], layers[1], llm_decoder
    Frozen: layers[2]~[7], speech_embedding, embed_tokens
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers 0 and 1 (which correspond to target layers 0 and 1)
    for i in range(2):
        for param in model.llm.model.model.layers[i].parameters():
            param.requires_grad = True

    # Unfreeze llm_decoder
    for param in model.llm_decoder.parameters():
        param.requires_grad = True

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Draft model: {}/{} params trainable ({:.1f}%)".format(trainable, total, 100 * trainable / total))


def get_args():
    parser = argparse.ArgumentParser(description="Train draft model for SSD")
    parser.add_argument(
        "--train_engine", default="torch_ddp", choices=["torch_ddp", "deepspeed"], help="Engine for paralleled training"
    )
    parser.add_argument("--model", required=True, help="model which will be trained")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument("--qwen_pretrain_path", required=False, help="qwen pretrain path")
    parser.add_argument("--onnx_path", required=False, help="onnx path")
    parser.add_argument("--checkpoint", help="checkpoint model (llm_draft.pt)")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="tensorboard log dir")
    parser.add_argument(
        "--ddp.dist_backend", dest="dist_backend", default="nccl", choices=["nccl", "gloo"], help="distributed backend"
    )
    parser.add_argument("--num_workers", default=0, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument(
        "--pin_memory", action="store_true", default=False, help="Use pinned memory buffers used for reading"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision training")
    parser.add_argument(
        "--deepspeed.save_states",
        dest="save_states",
        default="model_only",
        choices=["model_only", "model+optimizer"],
        help="save model/optimizer states",
    )
    parser.add_argument("--timeout", default=60, type=int, help="timeout (in seconds) of cosyvoice_join.")
    parser.add_argument("--wandb_project", default="cosyvoice-ssd", help="wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="wandb run name (auto-generated if not set)")
    if HAS_DEEPSPEED:
        parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    if args.onnx_path:
        os.environ["onnx_path"] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    # Set default distributed env vars for single-GPU training
    for key, default in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29500"),
                         ("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")]:
        if key not in os.environ:
            os.environ[key] = default

    # Build override dict — only override keys that exist in the config
    # Draft configs may not have flow/hift/hifigan sections
    override_dict = {}
    with open(args.config, "r") as f:
        config_text = f.read()
    for key in ["flow", "hift", "hifigan"]:
        # Check if key is a top-level YAML key (not indented, not a comment)
        if "\n{}: ".format(key) in config_text or "\n{}:".format(key) in config_text or config_text.startswith("{}:".format(key)):
            override_dict[key] = None
    if args.qwen_pretrain_path is not None:
        override_dict["qwen_pretrain_path"] = args.qwen_pretrain_path
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs["train_conf"].update(vars(args))

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(
        args, configs, gan=False, dpo=False
    )

    # Do some sanity checks and save config to args.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Load model
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            if "step" in state_dict:
                start_step = state_dict["step"]
            if "epoch" in state_dict:
                start_epoch = state_dict["epoch"]
            logging.info("Loaded checkpoint from {}".format(args.checkpoint))
        else:
            logging.warning("checkpoint {} does not exist!".format(args.checkpoint))

    # Apply freeze strategy
    freeze_draft_model(model)

    # Initialize wandb (rank 0 only)
    if HAS_WANDB and int(os.environ.get('RANK', 0)) == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model,
                "config_file": args.config,
                "train_engine": args.train_engine,
                "total_params": total_params,
                "trainable_params": trainable_params,
                **{k: v for k, v in configs["train_conf"].items()
                   if isinstance(v, (int, float, str, bool))},
            },
        )
        logging.info("wandb initialized: project={}, run={}".format(
            args.wandb_project, wandb.run.name))
    elif not HAS_WANDB:
        logging.warning("wandb not installed, skipping wandb logging")

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(
        args, configs, model, gan=False
    )
    scheduler.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs["train_conf"])
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    save_model(model, "init", info_dict)

    # Get executor
    executor = Executor(gan=False)
    executor.step = start_step

    # Init scaler for amp
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    logging.info("start step {} start epoch {}".format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(
            model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join
        )
        dist.destroy_process_group(group_join)

    # Finish wandb
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

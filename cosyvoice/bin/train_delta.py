# Copyright (c) 2026 DELTA-TTS reproduction for CosyVoice3
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

"""Train the DELTA-TTS delta conversion of CosyVoice3 (AR -> masked diffusion LM).

The pretrained AR CosyVoice3 llm.pt stays frozen; only the delta modules of
cosyvoice.llm.diffusion_llm.DiffusionCosyVoice3LM are trained
(docs/delta_tts_reproduction_plan.md, docs/delta_tts_phase0_verification.md):
  - LoRA adapters (r=64) on q/k/v/o/gate/up/down projections of all decoder layers
  - one ConformerConvModule per decoder layer (zero-initialized, identity at start)
  - the [M] mask embedding (initialized from the mean of speech_embedding)

Conversion order matters: the AR checkpoint is loaded BEFORE apply_lora() so its keys
still match the vanilla Qwen2 module names (peft renames q_proj -> q_proj.base_layer),
and init_mask_embedding() runs AFTER the load so it sees the pretrained
speech_embedding statistics.

Checkpoints: the regular full state_dict saves (train_utils.save_model) are kept, and
every save additionally writes a small `<name>_delta.pt` holding only the trainable
delta parameters (lora / conv / mask_emb). Resume from it with --delta_checkpoint on
top of the original --checkpoint llm.pt.

Usage:
    python cosyvoice/bin/train_delta.py \
        --train_engine torch_ddp --ddp.dist_backend gloo --model llm \
        --config examples/libritts/cosyvoice3/conf/cosyvoice3_delta.yaml \
        --train_data data/libritts/train.list --cv_data data/libritts/cv.list \
        --model_dir ./checkpoints_delta \
        --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
        --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
        --num_workers 1 --use_amp
"""

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist

# NOTE deepspeed/wandb are optional, training runs with torch_ddp alone
try:
    import deepspeed
except ImportError:
    deepspeed = None
try:
    import wandb
except ImportError:
    wandb = None

from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.lr_scheduler import _LRScheduler

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


class ConstantWithWarmupLR(_LRScheduler):
    """Linear warmup to base_lr, then constant (paper recipe: lr 1e-4 constant
    after a 2000-step linear warmup).

    cosyvoice.utils.scheduler has no such scheduler (WarmupLR decays as inverse
    sqrt after the warmup, ConstantLR has no warmup) and train_utils must stay
    unmodified, so main() swaps the WarmupLR built by init_optimizer_and_scheduler
    for this one. set_step()-compatible with the cosyvoice schedulers.
    """

    def __init__(self, optimizer, warmup_steps=2000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps <= 0:
            return list(self.base_lrs)
        return [lr * min(1.0, step_num / self.warmup_steps) for lr in self.base_lrs]

    def set_step(self, step: int):
        self.last_epoch = step


def save_delta_checkpoint(model, model_name, info_dict):
    """Save only the trainable delta parameters as {model_dir}/{model_name}_delta.pt.

    The file is a dict of state_dict buckets {'lora': ..., 'conv': ..., 'mask_emb': ...}
    plus 'epoch'/'step', a few hundred MB smaller than the full state_dict saves.
    """
    if int(os.environ.get('RANK', 0)) != 0:
        return
    m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    delta_state = {'lora': {}, 'conv': {}, 'mask_emb': {}}
    for name, param in m.named_parameters():
        if param.requires_grad is False:
            continue
        if 'lora_' in name:
            delta_state['lora'][name] = param.detach().cpu()
        elif 'conv_modules' in name:
            delta_state['conv'][name] = param.detach().cpu()
        elif 'mask_emb' in name:
            delta_state['mask_emb'][name] = param.detach().cpu()
        else:
            logging.warning('trainable parameter {} matches no delta bucket, not saved in delta checkpoint'.format(name))
    delta_state['epoch'] = info_dict['epoch']
    delta_state['step'] = info_dict['step']
    save_path = os.path.join(info_dict['model_dir'], '{}_delta.pt'.format(model_name))
    torch.save(delta_state, save_path)
    logging.info('[Rank 0] Delta checkpoint: save trainable-only state to {}'.format(save_path))


class DeltaExecutor(Executor):
    """Executor that writes a trainable-only `<name>_delta.pt` alongside every full checkpoint."""

    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        super().cv(model, cv_data_loader, writer, info_dict, on_batch_end)
        # keep the same naming as Executor.cv -> save_model
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_delta_checkpoint(model, model_name, info_dict)


def load_delta_checkpoint(model, delta_checkpoint):
    """Load a trainable-only delta checkpoint saved by save_delta_checkpoint. Call after conversion."""
    delta_state = torch.load(delta_checkpoint, map_location='cpu')
    flat_state_dict = {}
    for bucket in ['lora', 'conv', 'mask_emb']:
        flat_state_dict.update(delta_state.get(bucket, {}))
    missing, unexpected = model.load_state_dict(flat_state_dict, strict=False)
    if len(unexpected) > 0:
        logging.warning('unexpected keys in delta checkpoint {}: {}'.format(delta_checkpoint, unexpected))
    logging.info('loaded delta checkpoint {} ({} tensors)'.format(delta_checkpoint, len(flat_state_dict)))
    start_step = delta_state.get('step', 0)
    start_epoch = delta_state.get('epoch', -1)
    return start_step, start_epoch


def build_delta_model(args, configs):
    """Build DiffusionCosyVoice3LM and convert it for delta training.

    Order: load AR llm.pt (strict=False) -> apply_lora() -> attach_conv_modules()
    -> init_mask_embedding() -> [optional delta resume] -> freeze_for_delta().
    """
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logging.info('loaded AR checkpoint {} (missing {} unexpected {})'.format(args.checkpoint, len(missing), len(unexpected)))
            # guard: a delta-converted full save (epoch_X_whole.pt) carries renamed keys
            # (peft base_layer / lora_* / conv_modules.* / layers.N.layer.*) that all land
            # in `unexpected`, so it would silently load almost nothing here
            converted = [k for k in unexpected
                         if '.base_layer.' in k or 'lora_' in k or k.startswith('conv_modules.') or '.layer.' in k]
            if len(converted) > 0:
                raise RuntimeError(
                    '--checkpoint {} contains post-conversion keys (e.g. {}): it is a delta-converted full '
                    'checkpoint, not the pretrained AR llm.pt. Pass the AR llm.pt to --checkpoint and resume '
                    'delta training with --delta_checkpoint instead.'.format(args.checkpoint, converted[0]))
            non_delta_missing = [k for k in missing if k != 'mask_emb']
            if len(non_delta_missing) > 0:
                logging.warning('AR checkpoint {} left {} non-delta keys uninitialized, e.g. {}'.format(
                    args.checkpoint, len(non_delta_missing), non_delta_missing[:5]))
            # NOTE 'step'/'epoch' in the AR checkpoint are deliberately ignored: --checkpoint is
            # the frozen pretrained base, the resume position comes from --delta_checkpoint only
            # (inheriting them would skip the warmup and could silently skip training entirely)
        else:
            logging.warning('checkpoint {} does not exist!'.format(args.checkpoint))

    model.apply_lora()
    model.attach_conv_modules()
    # must run after the AR checkpoint load, uses the pretrained speech_embedding mean
    model.init_mask_embedding()

    if args.delta_checkpoint is not None:
        if os.path.exists(args.delta_checkpoint):
            start_step, start_epoch = load_delta_checkpoint(model, args.delta_checkpoint)
        else:
            logging.warning('delta checkpoint {} does not exist!'.format(args.delta_checkpoint))

    model.freeze_for_delta()
    summary = model.trainable_parameter_summary()
    logging.info('delta trainable parameter summary: {}'.format(summary))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Delta model: {}/{} params trainable ({:.2f}%)'.format(trainable, total, 100.0 * trainable / max(1, total)))
    return model, start_step, start_epoch


def get_args():
    parser = argparse.ArgumentParser(description='Train the DELTA-TTS delta conversion (masked diffusion) of CosyVoice3')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', default='llm', help='model which will be trained, only llm is supported here')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='pretrained AR checkpoint (llm.pt), loaded before the delta conversion')
    parser.add_argument('--delta_checkpoint', required=False,
                        help='trainable-only <name>_delta.pt to resume delta training from, loaded after the conversion')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training (bf16 with torch_ddp, per the paper recipe)')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--wandb_project', default='cosyvoice-delta', help='wandb project name')
    parser.add_argument('--wandb_run_name', default=None, help='wandb run name (auto-generated if not set)')
    if deepspeed is not None:
        parser = deepspeed.add_config_arguments(parser)
    else:
        # keep the recipe command lines working without deepspeed installed
        parser.add_argument('--deepspeed', action='store_true', default=False)
        parser.add_argument('--deepspeed_config', type=str, default=None)
    args = parser.parse_args()
    if args.train_engine == 'deepspeed' and deepspeed is None:
        raise ImportError('deepspeed is not installed but --train_engine deepspeed is requested')
    return args


@record
def main():
    args = get_args()
    # NOTE only set when given, the env var flips online_feature=True in cosyvoice.utils.onnx
    if args.onnx_path is not None:
        os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # default distributed env vars for single-node / single-GPU (or Windows) runs
    for key, default in [('MASTER_ADDR', 'localhost'), ('MASTER_PORT', '29500'),
                         ('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')]:
        if key not in os.environ:
            os.environ[key] = default

    # the delta yaml is llm-only, only override the sections that actually exist in it
    override_dict = {}
    with open(args.config, 'r') as f:
        config_text = f.read()
    for key in ['flow', 'hift', 'hifigan']:
        if '\n{}:'.format(key) in config_text or config_text.startswith('{}:'.format(key)):
            override_dict[key] = None
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    if torch.cuda.is_available():
        init_distributed(args)
    else:
        # init_distributed calls torch.cuda.set_device which fails on CPU-only machines,
        # init the (gloo) process group directly for CPU smoke runs
        logging.warning('CUDA not available, initializing process group without torch.cuda.set_device')
        dist.init_process_group(args.dist_backend)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan=False, dpo=False)

    # Do some sanity checks and save config to args.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Build model, load the AR checkpoint and apply the delta conversion + freeze
    model, start_step, start_epoch = build_delta_model(args, configs)

    # Initialize wandb (rank 0 only)
    if wandb is not None and int(os.environ.get('RANK', 0)) == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': args.model,
                'config_file': args.config,
                'train_engine': args.train_engine,
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                **{k: v for k, v in configs['train_conf'].items() if isinstance(v, (int, float, str, bool))},
            },
        )
        logging.info('wandb initialized: project={}, run={}'.format(args.wandb_project, wandb.run.name))
    elif wandb is None:
        logging.warning('wandb not installed, skipping wandb logging')

    # Dispatch model from cpu to gpu
    if torch.cuda.is_available():
        model = wrap_cuda_model(args, model)
    else:
        # batch_forward passes the LOCAL_RANK int as device; DiffusionCosyVoice3LM.forward
        # normalizes it to the model's own device when CUDA is unavailable, so training
        # steps do run on CPU (very slowly) instead of crashing on tensor.to(0)
        logging.warning('CUDA not available, keeping the model on CPU (slow, smoke test only)')

    # Get optimizer & scheduler; frozen params get no grad so passing model.parameters() is fine
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan=False)
    if args.train_engine == 'torch_ddp' and configs['train_conf']['scheduler'] == 'warmuplr':
        # paper recipe: lr stays constant at base_lr after the warmup. WarmupLR decays as
        # inverse sqrt past the warmup (1/10 of base_lr by step 200k), so replace it here
        # (train_utils only knows warmuplr / NoamHoldAnnealing / constantlr, none of which
        # implements warmup-then-constant, and existing files must not be modified)
        scheduler = ConstantWithWarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        logging.info('replaced WarmupLR with ConstantWithWarmupLR({}) to keep lr constant after warmup (paper recipe)'.format(
            configs['train_conf']['scheduler_conf']))
    scheduler.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)
    save_delta_checkpoint(model, 'init', info_dict)

    # Get executor
    executor = DeltaExecutor(gan=False)
    executor.step = start_step

    # NOTE loss scaling only makes sense for fp16, torch_ddp + --use_amp selects bf16 where scaler stays None
    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and configs['train_conf']['dtype'] == 'fp16') else None
    logging.info('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend='gloo', timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)

    if wandb is not None and int(os.environ.get('RANK', 0)) == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()

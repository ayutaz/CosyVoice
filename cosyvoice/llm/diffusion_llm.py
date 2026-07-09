# Copyright (c) 2026 DELTA-TTS reproduction (built on CosyVoice, Alibaba Inc)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DELTA-TTS: adapting the CosyVoice3 AR LM into a masked diffusion LM.

Implements the DELTA-TTS recipe (arXiv 2607.04140) on top of the frozen
CosyVoice3 backbone: LoRA on all attention/MLP projections, a Conformer-style
convolution module after every decoder layer, a trainable [M] embedding,
bidirectional attention via a 4D additive mask, the output-side shift that
preserves the pretrained AR contract (hidden[i] predicts token i+1), a
1/t-weighted masked CE training objective, and confidence-ordered parallel
decoding with the time-shifted unmasking schedule.

See docs/delta_tts_phase0_verification.md (B-1..B-4) for the verified
implementation contract this module follows.
"""
import math
import random
from typing import Callable, Dict, Generator, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from cosyvoice.llm.llm import CosyVoice3LM
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.utils.mask import make_pad_mask

# rule-based length fallback when no prompt is available: speech tokens per text token
FALLBACK_TOKEN_TEXT_RATIO = 6.0
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def build_bidirectional_mask(pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build a (B, 1, T, T) additive float attention mask for bidirectional attention.

    0.0 = visible, torch.finfo(dtype).min = blocked. Passing a 4D mask makes
    transformers bypass its internal causal/padding/sliding-window mask
    composition entirely (phase0 doc B-1), so the causal structure disappears
    and every valid position attends to every valid position.

    Padding is blocked on the key-side columns only. Query-side padding rows
    are deliberately NOT fully blocked: a row whose every key is at min would
    softmax to NaN on CPU/sdpa where _unmask_unattended does not apply.

    NOTE flash_attention_2 does not support 4D additive masks; the backbone
    must run with the default sdpa (or eager) attention implementation.

    Args:
        pad_mask: (B, T) bool, True = valid position.
        dtype: dtype of the returned mask, must match the attention dtype.
    Returns:
        (B, 1, T, T) additive float mask.
    """
    assert pad_mask.dtype == torch.bool, 'pad_mask must be bool with True = valid'
    batch, length = pad_mask.shape
    mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=pad_mask.device)
    mask.masked_fill_(~pad_mask[:, None, None, :], torch.finfo(dtype).min)
    return mask


def shift_align(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Align logits and targets under the AR output-side shift (phase0 doc B-2).

    The input side is never shifted ([M] sits at the masked position j itself);
    the pretrained AR contract "hidden[i] predicts token i+1" is kept, so the
    token at masked position j is read from the logits at position j-1. This is
    realized by dropping the last logits step and the first target/mask step.
    Position 0 (sos) is never masked, so dropping mask[:, 0] loses nothing.

    Args:
        logits: (B, T, V) float.
        targets: (B, T) long token ids (IGNORE_ID at positions never read).
        mask: (B, T) bool, True at positions whose input was replaced by [M].
    Returns:
        (sl_flat, tgt_flat): (N, V) logits and (N,) targets, one row per masked position.
    """
    sl = logits[:, :-1]
    tgt = targets[:, 1:]
    m = mask[:, 1:]
    return sl[m], tgt[m]


def unmask_schedule(L: int, T: int, mu: float) -> List[int]:
    """Tokens to commit at each of T decoding steps under the time-shifted schedule.

    c_n = mu*(n/T) / (1 + (mu-1)*(n/T)), k_n = floor(c_n*L) - floor(c_{n-1}*L).
    The final step is forced to commit every remaining masked token: in floating
    point c_T can land just below 1.0 (mu=0.3 gives c_T=0.9999999999999998),
    which would otherwise leave one token permanently masked (phase0 doc B-2).

    Returns a list of T non-negative ints whose sum is exactly L.
    """
    if T <= 0:
        raise ValueError('num_steps must be positive, got {}'.format(T))
    if L < 0:
        raise ValueError('target length must be non-negative, got {}'.format(L))
    if mu <= 0:
        raise ValueError('mu must be positive, got {}'.format(mu))
    ks, committed = [], 0
    for n in range(1, T + 1):
        if n == T:
            k = L - committed
        else:
            s = n / T
            c_n = mu * s / (1.0 + (mu - 1.0) * s)
            k = max(0, math.floor(c_n * L) - committed)
        ks.append(k)
        committed += k
    return ks


def _nucleus_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Vectorized top-p (nucleus) sampling over the rows of an (N, V) probability matrix.

    The existing cosyvoice.utils.common.nucleus_sampling is single-row and
    python-loop based; the diffusion decoder samples every masked position in
    parallel, hence this batched variant. The first (highest probability)
    token of each row is always kept, so multinomial never sees an all-zero row.

    Returns (N,) long sampled token ids.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    keep = (cum - sorted_probs) < top_p
    filtered = sorted_probs * keep
    choice = filtered.multinomial(1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


class ConformerConvModule(torch.nn.Module):
    """Conformer-style convolution module, one per decoder layer (DELTA-TTS Sec 3).

    LayerNorm -> pointwise Conv1d(d->2d) -> GLU -> depthwise Conv1d(k=31,
    groups=d) -> LayerNorm -> SiLU -> pointwise Conv1d(d->d) -> Dropout ->
    residual add.

    Design decisions (neither is specified by the paper):
    - LayerNorm is used where the canonical Conformer uses BatchNorm after the
      depthwise conv: BatchNorm statistics would be corrupted by padded
      positions in variable-length batches, LayerNorm is per-position and
      padding-safe (deliberate deviation from the canonical Conformer).
    - The last pointwise conv is zero-initialized so the whole module is an
      exact identity (pure residual passthrough) at training start, preserving
      the pretrained backbone behaviour before any delta training happens.

    pad_mask (bool, True = valid) zeroes padded positions right before the
    depthwise conv, the only op with a temporal receptive field, so padding
    values can never leak into valid positions.

    d=896 gives ~2.44M parameters per module; 24 modules ~= 58.6M, consistent
    with the paper's ~59M conv parameter budget.
    """

    def __init__(self, channels: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size must be odd to preserve sequence length'
        self.norm1 = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                        groups=channels, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.pointwise_conv2.weight)
        nn.init.zeros_(self.pointwise_conv2.bias)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, D); pad_mask: (B, T) bool with True = valid, or None (all valid)."""
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(1).to(x.dtype)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


class _ConvContext:
    """Shared mutable holder passing the per-forward pad mask to the layer wrappers.

    The HF decoder-layer call signature cannot carry a custom kwarg, so the
    owning DiffusionCosyVoice3LM sets pad_mask before each backbone call and
    resets it afterwards. Not thread-safe across concurrent forwards on the
    same module (the existing single-threaded llm_job usage is fine).
    """
    __slots__ = ['pad_mask']

    def __init__(self):
        self.pad_mask = None


class ConvWrappedDecoderLayer(torch.nn.Module):
    """Wraps one Qwen2 decoder layer: run the layer, apply the conv module to the
    output hidden states, return the same tuple shape.

    NOTE the wrapped layer is registered under the attribute name 'layer', so
    after attach_conv_modules() the backbone state_dict keys change from
    'llm.model.model.layers.N.<param>' to 'llm.model.model.layers.N.layer.<param>'.
    Load the pretrained checkpoint BEFORE attaching the conv modules.

    The conv module is deliberately kept as a plain (unregistered) reference:
    its parameters are owned by DiffusionCosyVoice3LM.conv_modules and would
    otherwise appear twice in the state_dict.
    """

    def __init__(self, layer: torch.nn.Module, conv: ConformerConvModule, ctx: _ConvContext):
        super().__init__()
        self.layer = layer
        # bypass nn.Module registration, see class docstring
        self.__dict__['conv'] = conv
        self.ctx = ctx

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.layer(hidden_states, *args, **kwargs)
        hidden = self.conv(outputs[0], self.ctx.pad_mask)
        return (hidden,) + outputs[1:]


class DiffusionCosyVoice3LM(CosyVoice3LM):
    """DELTA-TTS masked-diffusion LM built on the frozen CosyVoice3 AR backbone.

    Differences from CosyVoice3LM:
    - forward() is a masked-diffusion loss (1/t weighted CE on masked positions
      with the output-side shift) and inference_diffusion() replaces AR
      decoding. The inherited inference()/inference_bistream() dispatch on the
      class name string and would fail with a bare ValueError, so they are
      overridden to raise a descriptive NotImplementedError instead.
    - Attention is bidirectional via a 4D additive mask, which requires the
      sdpa or eager attention implementation (flash_attention_2 does not
      accept 4D additive masks).
    - The [M] mask embedding is an independent nn.Parameter (self.mask_emb),
      NOT an unused speech_embedding vocabulary slot: freeze_for_delta() must
      keep speech_embedding fully frozen while [M] stays trainable, and rows
      of an nn.Embedding cannot be frozen individually (intentional change
      from the spare-slot idea in phase0 doc C-1). Call init_mask_embedding()
      AFTER loading the pretrained checkpoint.

    Typical setup order (mirrored by cosyvoice/bin/train_delta.py):
        model = DiffusionCosyVoice3LM(...)
        model.load_state_dict(torch.load('llm.pt'), strict=False)
        model.apply_lora()
        model.attach_conv_modules()
        model.init_mask_embedding()
        model.freeze_for_delta()
    """

    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
            lora_r: int = 64,
            lora_alpha: int = 128,
            lora_dropout: float = 0.0,
            conv_kernel: int = 31,
            conv_dropout: float = 0.1,
            prompt_ratio_max: float = 0.5,
            prompt_drop: float = 0.1,
            t_min: float = 0.01,
            num_steps: int = 16,
            mu: float = 0.3,
            top_p: float = 0.8,
            length_scale: float = 1.0,
            tokens_per_mora: float = 4.708,
    ):
        super().__init__(llm_input_size, llm_output_size, speech_token_size, llm, sampling,
                         length_normalized_loss, lsm_weight, mix_ratio)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.conv_kernel = conv_kernel
        self.conv_dropout = conv_dropout
        self.prompt_ratio_max = prompt_ratio_max
        self.prompt_drop = prompt_drop
        self.t_min = t_min
        self.num_steps = num_steps
        self.mu = mu
        self.top_p = top_p
        self.length_scale = length_scale
        # 25Hz speech tokens per mora, calibration MEDIAN over the moe_speech train
        # shards (scripts/calibrate_tokens_per_mora.py). p60=5.0 was tried for headroom
        # and regressed both eval sets (fixed-length decoding suffers from surplus
        # budget too); consumed by the CLI mora-based length rule
        self.tokens_per_mora = tokens_per_mora
        # [M] embedding, independent parameter (see class docstring). Zeros here,
        # real init happens in init_mask_embedding() after checkpoint load.
        self.mask_emb = torch.nn.Parameter(torch.zeros(llm_input_size))
        self.conv_modules: Optional[torch.nn.ModuleList] = None
        self.peft_model = None
        self._conv_ctx = _ConvContext()

    def apply_lora(self):
        """Inject LoRA adapters into every q/k/v/o/gate/up/down projection.

        get_peft_model replaces the target submodules in place, so the direct
        backbone call path self.llm.model.model(...) picks the adapters up
        (verified in scripts/spikes/s3_lora_peft.py). The returned PeftModel is
        kept in self.peft_model for save_pretrained/disable_adapter/merge
        (phase0 doc B-3); it is stored via __dict__ to avoid registering it as
        a submodule, because it wraps self.llm.model (already registered) and
        normal attribute assignment would duplicate every backbone parameter
        in the state_dict.
        """
        from peft import LoraConfig, get_peft_model
        if self.peft_model is not None:
            raise RuntimeError('apply_lora() called twice')
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=self.lora_dropout,
            bias='none',
            task_type=None,
        )
        self.__dict__['peft_model'] = get_peft_model(self.llm.model, lora_config)
        return self.peft_model

    def attach_conv_modules(self):
        """Attach one ConformerConvModule after every Qwen2 decoder layer.

        The conv modules are owned by self.conv_modules (so they appear once in
        the state_dict and in parameters()); each decoder layer is replaced by a
        ConvWrappedDecoderLayer that shares the conv instance and reads the
        per-forward pad mask from self._conv_ctx. NOTE this nests the original
        layer parameters one level deeper in the state_dict (see
        ConvWrappedDecoderLayer docstring), so the pretrained checkpoint must be
        loaded before calling this.
        """
        if self.conv_modules is not None:
            raise RuntimeError('attach_conv_modules() called twice')
        layers = self.llm.model.model.layers
        self.conv_modules = torch.nn.ModuleList([
            ConformerConvModule(self.llm_input_size, kernel_size=self.conv_kernel, dropout=self.conv_dropout)
            for _ in range(len(layers))
        ])
        for i in range(len(layers)):
            layers[i] = ConvWrappedDecoderLayer(layers[i], self.conv_modules[i], self._conv_ctx)

    @torch.no_grad()
    def init_mask_embedding(self):
        """Initialize [M] to the mean of the real speech token embeddings.

        Must be called AFTER the pretrained checkpoint has been loaded,
        otherwise the mean is taken over randomly initialized rows.
        """
        self.mask_emb.copy_(self.speech_embedding.weight[:self.speech_token_size].mean(dim=0))

    def freeze_for_delta(self):
        """Freeze everything except the LoRA adapters, the conv modules and [M].

        llm_decoder, speech_embedding, the text embedding and the whole
        backbone stay frozen (the AR contract is preserved and the delta is
        absorbed by LoRA, phase0 doc B-2/B-3).
        """
        for param in self.parameters():
            param.requires_grad_(False)
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(True)
        if self.conv_modules is not None:
            for param in self.conv_modules.parameters():
                param.requires_grad_(True)
        self.mask_emb.requires_grad_(True)

    def trainable_parameter_summary(self) -> Dict[str, int]:
        """Count trainable parameters per group: lora / conv / mask_emb / other."""
        summary = {'lora': 0, 'conv': 0, 'mask_emb': 0, 'other': 0}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_' in name:
                summary['lora'] += param.numel()
            elif name.startswith('conv_modules.'):
                summary['conv'] += param.numel()
            elif name == 'mask_emb':
                summary['mask_emb'] += param.numel()
            else:
                summary['other'] += param.numel()
        return summary

    def load_delta_state(self, delta_checkpoint: str):
        """Load a trainable-only delta checkpoint (train_delta.save_delta_checkpoint format).

        The file holds {'lora': {...}, 'conv': {...}, 'mask_emb': {...}} state
        buckets plus 'epoch'/'step'. Call AFTER apply_lora()/attach_conv_modules()
        so the parameter names exist. Returns (step, epoch) for training resume.
        """
        delta_state = torch.load(delta_checkpoint, map_location='cpu')
        flat_state_dict = {}
        for bucket in ('lora', 'conv', 'mask_emb'):
            flat_state_dict.update(delta_state.get(bucket, {}))
        if len(flat_state_dict) == 0:
            raise RuntimeError('{} holds no lora/conv/mask_emb buckets, not a delta checkpoint'.format(delta_checkpoint))
        _, unexpected = self.load_state_dict(flat_state_dict, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError('delta checkpoint {} has keys the converted model does not: {} (was the '
                               'conversion order or lora/conv config changed?)'.format(delta_checkpoint, unexpected[:3]))
        return delta_state.get('step', 0), delta_state.get('epoch', -1)

    @classmethod
    def from_ar(cls, ar_llm: CosyVoice3LM, delta_checkpoint: Optional[str] = None, **delta_kwargs) -> 'DiffusionCosyVoice3LM':
        """Convert a weight-loaded AR CosyVoice3LM into a diffusion LM for inference.

        Intended for the CLI / evaluation flow where an AR model (pretrained or
        Japanese fine-tuned) is already built and loaded:

            cosyvoice = AutoModel(model_dir=...)
            cosyvoice.model.llm.load_state_dict(ja_state, strict=True)  # optional FT swap
            cosyvoice.model.llm = DiffusionCosyVoice3LM.from_ar(
                cosyvoice.model.llm, delta_checkpoint='.../epoch_X_delta.pt')

        CONSUMES ar_llm: the Qwen2 backbone module is reused by reference and
        LoRA/conv-wrapped in place, so the AR instance must not be used again.
        Weights are copied before attach_conv_modules() nests the backbone keys
        (same ordering rule as train_delta.build_delta_model). Without a
        delta_checkpoint the conversion is identity-preserving at first
        (zero-init LoRA-B / conv pointwise_conv2) but decoding quality requires
        a trained delta. The model is frozen and set to eval mode.
        """
        model = cls(
            llm_input_size=ar_llm.llm_input_size,
            llm_output_size=ar_llm.llm_output_size,
            speech_token_size=ar_llm.speech_token_size,
            llm=ar_llm.llm,
            sampling=ar_llm.sampling,
            **delta_kwargs,
        )
        missing, unexpected = model.load_state_dict(ar_llm.state_dict(), strict=False)
        assert missing == ['mask_emb'], 'unexpected missing keys converting from AR: {}'.format(missing)
        assert len(unexpected) == 0, 'AR state has keys the diffusion model does not: {}'.format(unexpected[:3])
        model.apply_lora()
        model.attach_conv_modules()
        model.init_mask_embedding()
        if delta_checkpoint is not None:
            model.load_delta_state(delta_checkpoint)
        model.freeze_for_delta()
        return model.eval()

    def _build_delta_sequence(self, text_emb: torch.Tensor, speech_emb: torch.Tensor, speech_token: torch.Tensor,
                              prompt_len: int, target_mask: torch.Tensor):
        """Build one unistream training sequence (S4: same-utterance prefix prompt).

        Layout: [sos, text(full transcript, never split), task_id,
        speech(first prompt_len tokens always visible, the rest masked per
        target_mask), eos]. sos/task_id/eos embeddings follow the CosyVoice3LM
        convention (speech_embedding rows 6561/6563/6562). Masked positions get
        self.mask_emb as input; token ids are kept at every speech position and
        the loss reads only the masked ones through the returned mask.

        Args:
            text_emb: (Lt, D) text-region embedding (optional instruct prefix
                followed by the full transcript, never split).
            speech_emb: (Ls, D) speech token embeddings.
            speech_token: (Ls,) long speech token ids.
            prompt_len: number of leading speech tokens kept visible (s_prompt).
            target_mask: (Ls - prompt_len,) bool, True = replace with [M].
        Returns:
            (seq_emb (L, D), token_ids (L,) long, seq_mask (L,) bool) with L = Lt + Ls + 3.
        """
        device = speech_emb.device
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, -1)
        eos_emb = self.speech_embedding.weight[self.eos_token].reshape(1, -1)
        speech_len = speech_emb.size(0)
        speech_mask = torch.zeros(speech_len, dtype=torch.bool, device=device)
        speech_mask[prompt_len:] = target_mask
        speech_in = torch.where(speech_mask.unsqueeze(-1), self.mask_emb.to(speech_emb.dtype).unsqueeze(0), speech_emb)
        seq_emb = torch.concat([sos_emb, text_emb, task_id_emb, speech_in, eos_emb], dim=0)
        offset = 2 + text_emb.size(0)
        token_ids = torch.full((seq_emb.size(0),), IGNORE_ID, dtype=torch.long, device=device)
        token_ids[offset:offset + speech_len] = speech_token
        token_ids[-1] = self.eos_token
        seq_mask = torch.zeros(seq_emb.size(0), dtype=torch.bool, device=device)
        seq_mask[offset:offset + speech_len] = speech_mask
        return seq_emb, token_ids, seq_mask

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Masked-diffusion training loss (phase0 doc B-2/B-4).

        Reads the same batch keys as Qwen2LM.forward: text_token(_len),
        speech_token(_len) and optionally instruct_token(_len). When
        instruct_token is present (e.g. the moe_speech recipe stores
        'You are a helpful assistant.<|endofprompt|>' per utterance) it is
        prepended to the text region, always visible and never in the loss:
        [sos, instruct, text, task_id, speech, eos]. This matches both the
        paper layout ([SOS, t_inst, ...]) and the unistream convention the
        AR backbone was (fine-)tuned with. At inference no separate instruct
        argument exists: the instruct text arrives inside the text tokens,
        as in the AR zero-shot/cross-lingual paths.

        Per sample: prompt prefix L_p = floor(u * Ls) with u ~ U(0,
        prompt_ratio_max), dropped to 0 with probability prompt_drop; masking
        ratio t ~ U(t_min, 1); the s_target region is masked iid Bernoulli(t).
        CE is computed on masked positions only via the output-side shift
        (shift_align), each sample weighted 1/t, and normalized by the total
        number of masked positions in the batch. acc follows the th_accuracy
        convention, restricted to masked positions.
        """
        # train_utils.batch_forward passes device as a bare int (the LOCAL_RANK CUDA
        # ordinal) and tensor.to(int) always means a CUDA device; resolve to the
        # model's own device on CUDA-less machines so CPU smoke training works.
        if not torch.cuda.is_available():
            device = next(self.parameters()).device
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        text_emb = self.llm.model.model.embed_tokens(text_token)
        speech_emb = self.speech_embedding(speech_token)
        instruct_emb, instruct_token_len = None, None
        if 'instruct_token' in batch:
            instruct_token_len = batch['instruct_token_len'].to(device)
            instruct_emb = self.llm.model.model.embed_tokens(batch['instruct_token'].to(device))

        seq_embs, token_ids, seq_masks, sample_weights = [], [], [], []
        for i in range(text_token.size(0)):
            text_len_i = int(text_token_len[i])
            speech_len_i = int(speech_token_len[i])
            prompt_len = int(random.random() * self.prompt_ratio_max * speech_len_i)
            if random.random() < self.prompt_drop:
                prompt_len = 0
            t = self.t_min + (1.0 - self.t_min) * random.random()
            target_mask = torch.rand(speech_len_i - prompt_len, device=device) < t
            text_region = text_emb[i, :text_len_i]
            if instruct_emb is not None:
                text_region = torch.concat([instruct_emb[i, :int(instruct_token_len[i])], text_region], dim=0)
            seq_emb, ids, mask = self._build_delta_sequence(text_region, speech_emb[i, :speech_len_i],
                                                            speech_token[i, :speech_len_i], prompt_len, target_mask)
            seq_embs.append(seq_emb)
            token_ids.append(ids)
            seq_masks.append(mask)
            sample_weights.append(1.0 / t)

        lm_input_len = torch.tensor([e.size(0) for e in seq_embs], dtype=torch.int32, device=device)
        lm_input = pad_sequence(seq_embs, batch_first=True, padding_value=0.0)
        lm_target = pad_sequence(token_ids, batch_first=True, padding_value=IGNORE_ID)
        lm_mask = pad_sequence(seq_masks, batch_first=True, padding_value=False)

        pad_mask = ~make_pad_mask(lm_input_len, lm_input.size(1))
        attn_mask = build_bidirectional_mask(pad_mask, lm_input.dtype)
        self._conv_ctx.pad_mask = pad_mask
        try:
            outs = self.llm.model.model(inputs_embeds=lm_input, attention_mask=attn_mask, use_cache=False)
        finally:
            self._conv_ctx.pad_mask = None
        logits = self.llm_decoder(outs.last_hidden_state)

        sl_flat, tgt_flat = shift_align(logits, lm_target, lm_mask)
        num_masked = tgt_flat.numel()
        if num_masked == 0:
            # degenerate batch: every Bernoulli draw came up visible. Keep the graph alive.
            loss = logits.sum() * 0.0
            acc = torch.tensor(0.0, device=logits.device)
            return {'loss': loss, 'acc': acc}
        weights = torch.tensor(sample_weights, dtype=torch.float32, device=device)
        weight_flat = weights.unsqueeze(1).expand(-1, lm_mask.size(1) - 1)[lm_mask[:, 1:]]
        ce = F.cross_entropy(sl_flat.float(), tgt_flat, reduction='none')
        loss = (weight_flat * ce).sum() / num_masked
        acc_target = lm_target[:, 1:].masked_fill(~lm_mask[:, 1:], IGNORE_ID)
        acc = th_accuracy(logits[:, :-1].reshape(-1, self.llm_decoder.out_features), acc_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def inference(self, *args, **kwargs):
        """Inherited AR decoding entry point, not available on the diffusion conversion.

        Qwen2LM.inference dispatches on the class name string and would raise a
        bare, message-less ValueError on the first generator iteration for this
        subclass (hard to diagnose inside llm_job threads); fail loudly instead.
        """
        raise NotImplementedError('DiffusionCosyVoice3LM has no AR decoding path, use inference_diffusion()')

    def inference_bistream(self, *args, **kwargs):
        """Inherited AR bistream decoding entry point, not available here (see inference)."""
        raise NotImplementedError('DiffusionCosyVoice3LM has no AR decoding path, use inference_diffusion()')

    @torch.inference_mode()
    def inference_diffusion(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
            num_steps: Optional[int] = None,
            mu: Optional[float] = None,
            top_p: Optional[float] = None,
            target_len: Optional[int] = None,
    ) -> Generator[int, None, None]:
        """Confidence-ordered parallel diffusion decoding (phase0 doc B-2).

        Signature-compatible with Qwen2LM.inference so CosyVoice3Model.llm_job
        can call either path; embedding, sampling, min_token_text_ratio and
        uuid are accepted but unused (max_token_text_ratio caps the rule-based
        length, see below). No CFG, no KV cache: every step is one
        full-sequence bidirectional forward (use_cache=False).

        prompt_text and text are concatenated into a single text region (S4:
        training never splits the text, so inference must not either).

        Rule-based target length (when target_len is None):
            r = len(prompt_speech_token) / max(1, len(prompt_text_tokens))
            target_len = ceil(r * len(text_tokens) * length_scale)
        The paper defines r per CHARACTER of the prompt transcript; this API
        only receives token ids, so the token count is used as a stand-in
        (deliberate deviation). For Japanese the CLI layer
        (cosyvoice/cli/cosyvoice.py) overrides this rule by passing an explicit
        mora-based target_len = ceil(mora_count(text) * tokens_per_mora *
        length_scale); it has to be computed there because the raw segment text
        never reaches this model layer — llm_job only forwards token ids, and
        the written form (kanji) diverges from the spoken length. The
        token-count rule below remains as the fallback when no mora count is
        available (non-Japanese text, streaming input). The ratio is only used
        when BOTH prompt
        sides are non-empty: without a prompt, and also when either side is
        empty (e.g. cross-lingual style calls passing a speech prompt without
        its transcript, where the ratio degenerates), a constant
        FALLBACK_TOKEN_TEXT_RATIO speech tokens per text token is assumed. The
        rule-based length is additionally capped at
        ceil(max_token_text_ratio * len(text_tokens)) — the AR-path envelope —
        so a pathological prompt (long audio, one-token transcript) cannot push
        it far outside the training length distribution; an explicitly passed
        target_len is never clamped.

        Decoding: all target positions start as [M]; each step runs a full
        forward, right-shifts the logits (position j reads hidden j-1), bans
        special tokens (>= speech_token_size), nucleus-samples every masked
        position, and commits the k_n most confident samples where k_n follows
        unmask_schedule (the final step commits all remaining masks). The
        generated tokens are yielded one by one (python ints) after the loop
        completes, matching the generator contract consumed by llm_job.
        """
        device = text.device
        num_steps = self.num_steps if num_steps is None else num_steps
        mu = self.mu if mu is None else mu
        top_p = self.top_p if top_p is None else top_p

        n_text = int(text_len)
        n_prompt_text = int(prompt_text_len)
        n_prompt_speech = int(prompt_speech_token_len)
        if target_len is None:
            if n_prompt_text > 0 and n_prompt_speech > 0:
                ratio = n_prompt_speech / max(1, n_prompt_text)
            else:
                # either prompt side empty: the ratio degenerates (see docstring)
                ratio = FALLBACK_TOKEN_TEXT_RATIO
            target_len = math.ceil(ratio * n_text * self.length_scale)
            target_len = min(target_len, math.ceil(max_token_text_ratio * n_text))
        target_len = max(int(target_len), 1)

        full_text = torch.concat([prompt_text, text], dim=1)
        text_emb = self.llm.model.model.embed_tokens(full_text)
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        eos_emb = self.speech_embedding.weight[self.eos_token].reshape(1, 1, -1)
        if n_prompt_speech > 0:
            prompt_speech_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype, device=device)
        prefix = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)
        offset = prefix.size(1)
        total_len = offset + target_len + 1

        attn_mask = build_bidirectional_mask(torch.ones(1, total_len, dtype=torch.bool, device=device), prefix.dtype)
        tokens = torch.full((target_len,), -1, dtype=torch.long, device=device)
        mask_emb = self.mask_emb.to(prefix.dtype).reshape(1, 1, -1)

        self._conv_ctx.pad_mask = None  # B=1, no padding: conv modules see every position as valid
        for k_n in unmask_schedule(target_len, num_steps, mu):
            if k_n <= 0:
                # nothing was committed since the previous step, the input sequence is
                # unchanged and the forward result would be discarded: skip the whole
                # full-sequence forward (strictly equivalent, no RNG is consumed here)
                continue
            committed = tokens >= 0
            # k_n > 0 implies remaining masks >= k_n (unmask_schedule sums to target_len)
            masked_pos = (~committed).nonzero(as_tuple=True)[0]
            target_emb = torch.where(committed.reshape(1, -1, 1),
                                     self.speech_embedding(tokens.clamp(min=0)).unsqueeze(0),
                                     mask_emb)
            lm_input = torch.concat([prefix, target_emb, eos_emb], dim=1)
            outs = self.llm.model.model(inputs_embeds=lm_input, attention_mask=attn_mask, use_cache=False)
            logits = self.llm_decoder(outs.last_hidden_state)
            # right shift: position j reads the logits produced by hidden j-1 (AR contract).
            # position 0 (sos) is never masked, so the duplicated first step is never read.
            logits = torch.concat([logits[:, :1], logits[:, :-1]], dim=1)
            pos_logits = logits[0, offset + masked_pos]
            pos_logits[:, self.speech_token_size:] = -float('inf')  # speech tokens only, no sos/eos/task/fill
            probs = pos_logits.softmax(dim=-1)
            sampled = _nucleus_sample(probs, top_p)
            # confidence = probability of the sampled token under the restricted softmax
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            k = min(int(k_n), masked_pos.numel())
            top_idx = confidence.topk(k).indices
            tokens[masked_pos[top_idx]] = sampled[top_idx]
        assert bool((tokens >= 0).all()), 'diffusion decode left masked positions, unmask_schedule is inconsistent'
        for token in tokens.tolist():
            yield token

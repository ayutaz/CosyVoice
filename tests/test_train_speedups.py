# Tests for the training speed optimizations: they must not change numerics.
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestQwen2EncoderBackbone:
    """Qwen2Encoder.forward now calls the Qwen2Model backbone directly to skip the
    unused full-vocab lm_head projection. The hidden states must be bit-identical."""

    def _tiny_model(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128)
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    def test_backbone_equals_hidden_states(self):
        model = self._tiny_model()
        x = torch.randn(2, 7, 32)
        masks = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
        with torch.no_grad():
            ref = model(inputs_embeds=x, attention_mask=masks,
                        output_hidden_states=True, return_dict=True).hidden_states[-1]
            out = model.model(inputs_embeds=x, attention_mask=masks, use_cache=False).last_hidden_state
        assert torch.equal(ref, out)

    def test_qwen2_encoder_integration(self, tmp_path):
        from cosyvoice.llm.llm import Qwen2Encoder
        model = self._tiny_model()
        model.save_pretrained(tmp_path)
        encoder = Qwen2Encoder(str(tmp_path))
        encoder.eval()
        x = torch.randn(2, 7, 32)
        xs_lens = torch.tensor([7, 5])
        with torch.no_grad():
            out, masks = encoder(x, xs_lens)
            ref = encoder.model(inputs_embeds=x, attention_mask=masks.squeeze(1),
                                output_hidden_states=True, return_dict=True).hidden_states[-1]
        assert torch.equal(ref, out)
        assert masks.shape == (2, 1, 7)


class TestLabelSmoothingFastPath:
    """smoothing == 0 now short-circuits into F.cross_entropy, it must match the
    original KLDiv formulation in value and gradient."""

    @staticmethod
    def _old_kl_loss(x, target, size, padding_idx, normalize_length):
        # original implementation, smoothing = 0
        criterion = torch.nn.KLDivLoss(reduction='none')
        batch_size = x.size(0)
        x = x.view(-1, size)
        target = target.view(-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(0.0)
        ignore = target == padding_idx
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), 1.0)
        kl = criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

    @pytest.mark.parametrize('normalize_length', [True, False])
    def test_matches_old_path(self, normalize_length):
        from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
        torch.manual_seed(0)
        size, padding_idx = 17, -1
        loss_fn = LabelSmoothingLoss(size=size, padding_idx=padding_idx, smoothing=0.0,
                                     normalize_length=normalize_length)
        x_new = torch.randn(3, 5, size, requires_grad=True)
        x_old = x_new.detach().clone().requires_grad_(True)
        target = torch.randint(0, size, (3, 5))
        target[0, 3:] = padding_idx
        target[2, 1:] = padding_idx

        new = loss_fn(x_new, target)
        old = self._old_kl_loss(x_old, target.clone(), size, padding_idx, normalize_length)
        assert torch.allclose(new, old, atol=1e-6)

        new.backward()
        old.backward()
        assert torch.allclose(x_new.grad, x_old.grad, atol=1e-6)

    def test_smoothing_path_unchanged(self):
        from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
        torch.manual_seed(0)
        loss_fn = LabelSmoothingLoss(size=17, padding_idx=-1, smoothing=0.1, normalize_length=True)
        x = torch.randn(2, 4, 17)
        target = torch.randint(0, 17, (2, 4))
        loss = loss_fn(x, target)
        assert torch.isfinite(loss)


class TestAudioLoad:
    """audio_load/audio_save replace torchaudio I/O, which requires the optional
    torchcodec package on torchaudio 2.9+."""

    def test_round_trip_path(self, tmp_path):
        from cosyvoice.utils.file_utils import audio_load, audio_save
        sr = 44100
        speech = torch.sin(2 * 3.14159 * 220 * torch.arange(sr) / sr).unsqueeze(0) * 0.5
        path = str(tmp_path / 'a.wav')
        audio_save(path, speech, sr)
        loaded, loaded_sr = audio_load(path)
        assert loaded_sr == sr
        assert loaded.shape == speech.shape
        assert loaded.dtype == torch.float32
        assert torch.allclose(loaded, speech, atol=1e-3)

    def test_bytesio_like_parquet_audio_data(self, tmp_path):
        from io import BytesIO
        from cosyvoice.utils.file_utils import audio_load, audio_save
        path = str(tmp_path / 'b.wav')
        audio_save(path, torch.zeros(1, 1600), 16000)
        with open(path, 'rb') as f:
            raw = f.read()
        loaded, sr = audio_load(BytesIO(raw))
        assert sr == 16000
        assert loaded.shape == (1, 1600)

    def test_load_wav_resample(self, tmp_path):
        from cosyvoice.utils.file_utils import audio_save, load_wav
        path = str(tmp_path / 'c.wav')
        audio_save(path, torch.zeros(2, 44100), 44100)  # stereo in
        speech = load_wav(path, 24000)
        assert speech.shape[0] == 1  # mono mixdown
        assert speech.shape[1] == 24000


class TestLlmDataPipeline:
    """waveform-free llm processors: filter/sort/batch/padding keyed on speech tokens."""

    @staticmethod
    def _sample(utt, n_speech, n_text=5):
        return {
            'utt': utt,
            'text': 'テスト文章です',
            'text_token': list(range(1, n_text + 1)),
            'instruct_token': [7, 8, 9],
            'speech_token': list(range(n_speech)),
            'utt_embedding': torch.randn(8),
            'spk_embedding': torch.randn(8),
        }

    def test_filter_speech_token(self):
        from cosyvoice.dataset.processor import filter_speech_token
        samples = [
            self._sample('keep', 100),
            self._sample('empty', 0),
            self._sample('too_short', 10),    # 40 x 10ms frames < min_length 100
            self._sample('too_long', 2000),   # 8000 frames > max_length 6000
        ]
        kept = list(filter_speech_token(iter(samples), max_length=6000, min_length=100))
        assert [s['utt'] for s in kept] == ['keep']

    def test_sort_and_dynamic_batch(self):
        from cosyvoice.dataset.processor import sort_by_speech_token, dynamic_batch_llm
        torch.manual_seed(0)
        samples = [self._sample('u{}'.format(i), n) for i, n in enumerate([50, 30, 90, 60, 20])]
        sorted_samples = list(sort_by_speech_token(iter(samples), sort_size=3))
        batches = list(dynamic_batch_llm(iter(sorted_samples), max_frames_in_batch=300))
        assert sum(len(b) for b in batches) == len(samples)
        for b in batches:
            longest = max(len(s['speech_token']) for s in b)
            assert longest * 2 * len(b) <= 300 or len(b) == 1

    def test_parquet_to_batch_end_to_end(self, tmp_path):
        """synthetic parquet -> column-pruned opener -> tokenize -> filter -> batch -> padding_llm"""
        import pandas as pd
        from cosyvoice.dataset.processor import (parquet_opener, tokenize, filter_speech_token,
                                                 parse_embedding, sort_by_speech_token,
                                                 dynamic_batch_llm, padding_llm)

        df = pd.DataFrame({
            'utt': ['spk1_{:03d}'.format(i) for i in range(6)],
            'audio_data': [b'\x00' * 1000] * 6,  # must be pruned by the columns arg
            'wav': ['/x/{}.wav'.format(i) for i in range(6)],
            'text': ['こんにちは世界{}'.format(i) for i in range(6)],
            'spk': ['spk1'] * 6,
            'utt_embedding': [[0.1] * 8] * 6,
            'spk_embedding': [[0.2] * 8] * 6,
            'speech_token': [list(range(40 + 10 * i)) for i in range(6)],
            'instruct': ['You are a helpful assistant.<|endofprompt|>'] * 6,
        })
        parquet_file = str(tmp_path / 'part.tar')
        df.to_parquet(parquet_file)

        class FakeTokenizer:
            def encode(self, text, allowed_special=None):
                return [ord(c) % 100 for c in text]

        columns = ['utt', 'text', 'instruct', 'utt_embedding', 'spk_embedding', 'speech_token']
        data = parquet_opener(iter([{'src': parquet_file}]), columns=columns)
        data = tokenize(data, get_tokenizer=FakeTokenizer, allowed_special='all')
        data = filter_speech_token(data, max_length=6000, min_length=100)
        data = parse_embedding(data, normalize=True)
        data = sort_by_speech_token(data, sort_size=4)
        data = dynamic_batch_llm(data, max_frames_in_batch=400)
        batches = list(padding_llm(data, use_spk_embedding=False))

        total = sum(len(b['utts']) for b in batches)
        assert total == 6
        for b in batches:
            assert 'audio_data' not in b
            assert set(['utts', 'text', 'text_token', 'text_token_len', 'speech_token',
                        'speech_token_len', 'instruct_token', 'instruct_token_len',
                        'utt_embedding', 'spk_embedding', 'embedding']) <= set(b.keys())
            assert b['speech_token'].dtype == torch.int64
            # embeddings normalized
            assert torch.allclose(b['utt_embedding'].norm(dim=1), torch.ones(len(b['utts'])), atol=1e-5)

    def test_padding_llm(self):
        from cosyvoice.dataset.processor import padding_llm
        batch_in = [self._sample('a', 10), self._sample('b', 30), self._sample('c', 20)]
        batch = next(padding_llm(iter([batch_in]), use_spk_embedding=False))
        # descending speech token length order
        assert batch['utts'] == ['b', 'c', 'a']
        assert batch['speech_token'].shape == (3, 30)
        assert batch['speech_token_len'].tolist() == [30, 20, 10]
        assert batch['text_token'].shape[0] == 3
        assert batch['instruct_token'].shape == (3, 3)
        assert torch.equal(batch['embedding'], batch['utt_embedding'])
        for key in ['speech_feat', 'whisper_feat', 'speech']:
            assert key not in batch

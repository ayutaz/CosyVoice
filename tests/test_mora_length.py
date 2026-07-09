# Tests for the mora-based DELTA-TTS target length rule: ja_frontend.mora_count,
# the CLI helper _diffusion_target_len and its wiring through model.tts / llm_job.
# Expected mora counts were pinned against the actual pyopenjtalk readings
# (e.g. ガッコウ -> ガッコー, 二分半です -> ニブン ハン デス).
import inspect
import json
import math
import os
import re
import sys

import pytest
import torch

from cosyvoice.cli.cosyvoice import CosyVoice2, _diffusion_target_len
from cosyvoice.cli.model import CosyVoice3Model
from cosyvoice.llm.diffusion_llm import DiffusionCosyVoice3LM
from cosyvoice.utils.ja_frontend import mora_count

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
TOKENS_PER_MORA = 4.708  # calibration median, scripts/calibration_tokens_per_mora.json
JA_TEXT = 'ニフンハン'  # 5 morae, the report's canonical example (二分半: 3 written chars)
EXPECTED_LEN = math.ceil(5 * TOKENS_PER_MORA)  # 25


class TestMoraCount:
    """contract: normal kana = 1, small youon kana = 0, sokuon/choon/N = 1, space/punctuation = 0."""

    @pytest.mark.parametrize('text,expected', [
        ('ニフンハン', 5),   # plain kana; reading 'ニ フン ハン', separator spaces count 0
        ('キョウ', 2),       # youon: small ョ merges into キ and counts 0
        ('ガッコウ', 4),     # sokuon ッ = 1; pyopenjtalk reads ガッコー, the choon = 1
        ('トーキョー', 4),   # long vowel mark = 1
        ('フェア', 2),       # small vowel kana ェ merges and counts 0
        ('シャッター', 4),   # youon + sokuon + choon combined
        ('', 0),
        ('You are', 0),      # non-japanese gate: ungated pyopenjtalk would spell 'ユー エーアールイー'
        ('hello world', 0),
    ])
    def test_counts(self, text, expected):
        assert mora_count(text) == expected

    def test_kanji_counts_by_reading_not_written_length(self):
        # 7 written chars -> reading ガッコー エ イッ タ = 8 morae
        assert mora_count('学校へ行った。') == 8
        # 二分半 in kana context reads ニブン ハン = 5 morae + デス = 7
        assert mora_count('二分半です') == 7
        assert mora_count('これは本です。') == 7  # コレ ワ ホン デス

    def test_space_and_punctuation_count_zero(self):
        # pre-normalized katakana input (example.py format) passes through unchanged
        assert mora_count('コレ ワ ホン デス。') == 7

    def test_instruct_prefix_is_ignored(self):
        text = 'You are a helpful assistant.<|endofprompt|>学校へ行った。'
        assert mora_count(text) == mora_count('学校へ行った。') == 8

    def test_instruct_prefix_with_non_japanese_payload(self):
        # only the last segment is spoken: a japanese instruct must not leak into the count
        assert mora_count('指示です。<|endofprompt|>hello') == 0

    def test_kanji_only_text_returns_zero(self):
        # documented gate limitation: kanji-only text is indistinguishable from
        # chinese, mora_count returns 0 and the caller falls back to the token rule
        assert mora_count('二分半') == 0

    @pytest.mark.parametrize('text,expected', [
        ('コンニチワ[breath]ミナサン', 9),      # [..] control tokens are not spoken and count 0;
        ('[breath]コンニチワ', 5),              # unstripped, pyopenjtalk would spell 'breath' letter-by-letter
        ('ヨロシク[laughter]', 4),
        ('コンニチワ<|breath|>ゲンキデス', 10),  # CV3-style <|..|> special tokens likewise
        ('[breath]', 0),                        # nothing left after stripping: fall back
        ('<|breath|>', 0),
    ])
    def test_inline_control_tokens_count_zero(self, text, expected):
        assert mora_count(text) == expected

    def test_missing_pyopenjtalk_raises_instead_of_silent_zero(self, monkeypatch):
        # a missing pyopenjtalk must fail loudly: silently returning 0 would disable
        # the whole mora rule (every call falls back to the token-ratio rule) with no trace
        monkeypatch.setitem(sys.modules, 'pyopenjtalk', None)
        with pytest.raises(ImportError):
            mora_count('学校へ行った。')


class _FakeDiffusionLLM:
    """attribute-level stand-in: hasattr(inference_diffusion) drives the CLI dispatch."""
    tokens_per_mora = TOKENS_PER_MORA
    length_scale = 1.0

    def inference_diffusion(self, **kwargs):
        raise NotImplementedError('CLI-layer tests never reach the model')


class _FakeARLLM:
    """no inference_diffusion attribute: the CLI must compute no target_len."""


class TestDiffusionTargetLen:
    """_diffusion_target_len = ceil(mora * tokens_per_mora * length_scale), else None."""

    def test_japanese_text_gives_mora_based_len(self):
        assert _diffusion_target_len(_FakeDiffusionLLM(), JA_TEXT) == EXPECTED_LEN

    def test_length_scale_multiplies(self):
        llm = _FakeDiffusionLLM()
        llm.length_scale = 1.2
        assert _diffusion_target_len(llm, JA_TEXT) == math.ceil(5 * TOKENS_PER_MORA * 1.2)  # 30

    def test_ar_llm_returns_none(self):
        assert _diffusion_target_len(_FakeARLLM(), JA_TEXT) is None

    def test_generator_text_returns_none(self):
        # streaming input arrives as a generator, the mora rule cannot apply
        assert _diffusion_target_len(_FakeDiffusionLLM(), (c for c in JA_TEXT)) is None

    def test_text_without_morae_returns_none(self):
        assert _diffusion_target_len(_FakeDiffusionLLM(), 'This is english.') is None
        assert _diffusion_target_len(_FakeDiffusionLLM(), '') is None
        assert _diffusion_target_len(_FakeDiffusionLLM(), '二分半') is None  # kanji-only, gate returns 0


class TestCalibrationConstantConsistency:
    """code default, yaml configs and the calibration artifact must agree."""

    YAMLS = [
        'examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta.yaml',
        'examples/moe_speech/cosyvoice3/conf/cosyvoice3_delta_refine.yaml',
    ]

    def test_code_default_is_calibrated_value(self):
        default = inspect.signature(DiffusionCosyVoice3LM.__init__).parameters['tokens_per_mora'].default
        assert default == TOKENS_PER_MORA

    def test_yaml_values_match_code_default(self):
        default = inspect.signature(DiffusionCosyVoice3LM.__init__).parameters['tokens_per_mora'].default
        for rel in self.YAMLS:
            with open(os.path.join(REPO_ROOT, rel), encoding='utf-8') as f:
                match = re.search(r'^tokens_per_mora:\s*([0-9.]+)', f.read(), re.M)
            assert match is not None, rel
            assert float(match.group(1)) == default, rel

    def test_calibration_json_matches(self):
        path = os.path.join(REPO_ROOT, 'scripts', 'calibration_tokens_per_mora.json')
        if not os.path.exists(path):
            pytest.skip('calibration artifact not present in this checkout')
        with open(path, encoding='utf-8') as f:
            result = json.load(f)
        assert result['tokens_per_mora'] == TOKENS_PER_MORA
        assert result['n'] >= 5000  # contract: calibrate on at least 5,000 utterances


class _RecordingModel:
    """stands in for CosyVoice2Model at the CLI layer, recording every tts kwarg set."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def tts(self, **kwargs):
        self.calls.append(kwargs)
        yield {'tts_speech': torch.zeros(1, 24000)}


class _FakeFrontend:
    """text passes through normalization untouched, model_input carries no tensors."""

    def text_normalize(self, text, split=True, text_frontend=True):
        return [text] if split else text

    def frontend_sft(self, *args):
        return {}

    def frontend_zero_shot(self, *args):
        return {}

    def frontend_cross_lingual(self, *args):
        return {}

    def frontend_instruct2(self, *args):
        return {}


def _make_cli(llm):
    cli = CosyVoice2.__new__(CosyVoice2)  # bypass __init__: no model dir on the test box
    cli.frontend = _FakeFrontend()
    cli.model = _RecordingModel(llm)
    cli.sample_rate = 24000
    return cli


CLI_METHODS = ['inference_sft', 'inference_zero_shot', 'inference_cross_lingual', 'inference_instruct2']


def _run_cli(cli, method, text=JA_TEXT):
    if method == 'inference_sft':
        return list(cli.inference_sft(text, 'spk'))
    if method == 'inference_zero_shot':
        return list(cli.inference_zero_shot(text, 'プロ', None))
    if method == 'inference_cross_lingual':
        return list(cli.inference_cross_lingual(text, None))
    if method == 'inference_instruct2':
        return list(cli.inference_instruct2(text, '指示', None))
    raise AssertionError('unknown method {}'.format(method))


class TestCLIWiring:
    """every text-bearing inference_* hands target_len to model.tts: a mora-based int
    for diffusion llms on japanese text, None otherwise (llm_job then keeps it away
    from the AR .inference path)."""

    @pytest.mark.parametrize('method', CLI_METHODS)
    def test_diffusion_llm_gets_mora_target_len(self, method):
        cli = _make_cli(_FakeDiffusionLLM())
        out = _run_cli(cli, method)
        assert len(out) == 1
        assert len(cli.model.calls) == 1
        assert cli.model.calls[0]['target_len'] == EXPECTED_LEN

    @pytest.mark.parametrize('method', CLI_METHODS)
    def test_ar_llm_gets_none(self, method):
        cli = _make_cli(_FakeARLLM())
        _run_cli(cli, method)
        assert cli.model.calls[0]['target_len'] is None

    def test_diffusion_llm_non_japanese_text_gets_none(self):
        cli = _make_cli(_FakeDiffusionLLM())
        _run_cli(cli, 'inference_cross_lingual', text='This is english.')
        assert cli.model.calls[0]['target_len'] is None

    def test_each_split_segment_gets_its_own_len(self):
        # target_len must be computed per synthesized segment, not per input text
        cli = _make_cli(_FakeDiffusionLLM())
        cli.frontend.text_normalize = lambda text, split=True, text_frontend=True: (['ニフンハン', 'キョウ'] if split else text)
        _run_cli(cli, 'inference_cross_lingual')
        assert [c['target_len'] for c in cli.model.calls] == [EXPECTED_LEN, math.ceil(2 * TOKENS_PER_MORA)]


class _RecordingDiffusionLLM:
    tokens_per_mora = TOKENS_PER_MORA
    length_scale = 1.0

    def __init__(self):
        self.calls = []

    def inference_diffusion(self, **kwargs):
        self.calls.append(kwargs)
        yield from [3, 4, 5]


class _RecordingARLLM:
    def __init__(self):
        self.calls = []

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        yield from [5, 7]


def _make_wrapper(llm):
    wrapper = CosyVoice3Model(llm=llm, flow=torch.nn.Identity(), hift=torch.nn.Identity(), fp16=False)
    wrapper.device = torch.device('cpu')  # keep the mock tensors on CPU even when the box has a GPU
    wrapper.silent_tokens = []  # keep the yielded counts deterministic
    return wrapper


def _job_args():
    return dict(text=torch.randint(0, 10, (1, 4)),
                prompt_text=torch.zeros(1, 0, dtype=torch.long),
                llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.long),
                llm_embedding=torch.zeros(1, 0))


class TestLLMJobTargetLen:
    """llm_job forwards target_len to inference_diffusion only; AR .inference never sees it."""

    def test_diffusion_receives_target_len(self):
        llm = _RecordingDiffusionLLM()
        wrapper = _make_wrapper(llm)
        uid = 'uid-diff'
        wrapper.tts_speech_token_dict[uid], wrapper.llm_end_dict[uid] = [], False
        wrapper.llm_job(uuid=uid, target_len=17, **_job_args())
        assert llm.calls[0]['target_len'] == 17
        assert wrapper.tts_speech_token_dict[uid] == [3, 4, 5]
        assert wrapper.llm_end_dict[uid] is True

    def test_positional_call_without_target_len_regression(self):
        # pre-change positional call signature (5 args) must keep working: default None
        llm = _RecordingDiffusionLLM()
        wrapper = _make_wrapper(llm)
        uid = 'uid-none'
        wrapper.tts_speech_token_dict[uid], wrapper.llm_end_dict[uid] = [], False
        args = _job_args()
        wrapper.llm_job(args['text'], args['prompt_text'], args['llm_prompt_speech_token'], args['llm_embedding'], uid)
        assert llm.calls[0]['target_len'] is None
        assert wrapper.tts_speech_token_dict[uid] == [3, 4, 5]
        assert wrapper.llm_end_dict[uid] is True

    def test_ar_inference_never_gets_target_len(self):
        # even a (misuse) target_len must not reach the AR signature, which lacks the kwarg
        llm = _RecordingARLLM()
        wrapper = _make_wrapper(llm)
        uid = 'uid-ar'
        wrapper.tts_speech_token_dict[uid], wrapper.llm_end_dict[uid] = [], False
        wrapper.llm_job(uuid=uid, target_len=17, **_job_args())
        assert 'target_len' not in llm.calls[0]
        assert wrapper.tts_speech_token_dict[uid] == [5, 7]
        assert wrapper.llm_end_dict[uid] is True


class TestTTSTargetLenWiring:
    """CosyVoice2/3Model.tts hands kwargs['target_len'] to llm_job (6th positional arg)."""

    def _run_tts(self, **tts_kwargs):
        wrapper = _make_wrapper(_RecordingDiffusionLLM())
        received = []

        def fake_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid, target_len=None):
            received.append(target_len)
            wrapper.tts_speech_token_dict[uuid].extend([1, 2, 3])
            wrapper.llm_end_dict[uuid] = True

        wrapper.llm_job = fake_llm_job
        wrapper.token2wav = lambda **kwargs: torch.zeros(1, 10)
        out = list(wrapper.tts(text=torch.zeros(1, 4, dtype=torch.int32), **tts_kwargs))
        assert len(out) == 1
        return received

    def test_target_len_reaches_llm_job(self):
        assert self._run_tts(target_len=17) == [17]

    def test_tts_without_target_len_backcompat(self):
        assert self._run_tts() == [None]

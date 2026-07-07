import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, 'examples', 'moe_speech', 'cosyvoice3', 'local', 'prepare_data.py')

spec = importlib.util.spec_from_file_location('prepare_moe_speech', MODULE_PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

# metadata example taken from the moe-speech-plus README
META_OK = {
    'parakeet_jp_transcription': '昨夜からずっと気配を探られていたか。',
    'anime_whisper_transcription': '昨夜からずっと気配を探られていたか…',
    'duration': 3.72,
    'speechMOS': 3.1,
}


def test_transcription_cer_ignores_style():
    # same content, only punctuation style differs
    assert prepare.transcription_cer('昨夜からずっと気配を探られていたか。', '昨夜からずっと気配を探られていたか…') == 0.0
    assert prepare.transcription_cer('こんにちは。', 'こんにちは！') == 0.0


def test_transcription_cer_detects_mismatch():
    assert prepare.transcription_cer('今日は良い天気です。', '全然違う内容の文章になっている。') > 0.5
    assert prepare.transcription_cer('', 'なにか') == 1.0


def test_filter_utt_accepts_good_utterance():
    text, reason = prepare.filter_utt(dict(META_OK))
    assert reason is None
    assert text == '昨夜からずっと気配を探られていたか。'


def test_filter_utt_transcription_choice():
    text, _ = prepare.filter_utt(dict(META_OK), transcription='anime_whisper')
    assert text == '昨夜からずっと気配を探られていたか…'


def test_filter_utt_rejects_low_mos():
    meta = dict(META_OK, speechMOS=1.5)
    text, reason = prepare.filter_utt(meta, mos_threshold=2.5)
    assert text is None and reason == 'mos'


def test_filter_utt_rejects_bad_duration():
    text, reason = prepare.filter_utt(dict(META_OK, duration=0.4))
    assert text is None and reason == 'duration'
    text, reason = prepare.filter_utt(dict(META_OK, duration=35.0))
    assert text is None and reason == 'duration'


def test_filter_utt_rejects_empty_text():
    text, reason = prepare.filter_utt(dict(META_OK, parakeet_jp_transcription=''))
    assert text is None and reason == 'empty_text'


def test_filter_utt_rejects_transcription_disagreement():
    meta = dict(META_OK, anime_whisper_transcription='全然違う内容の文章になっている。')
    text, reason = prepare.filter_utt(meta, max_cross_cer=0.2)
    assert text is None and reason == 'cross_cer'
    # disabled with -1
    text, reason = prepare.filter_utt(meta, max_cross_cer=-1)
    assert reason is None


def test_filter_utt_tolerates_missing_metrics():
    text, reason = prepare.filter_utt({'parakeet_jp_transcription': 'こんにちは。', 'anime_whisper_transcription': ''})
    assert reason is None and text == 'こんにちは。'


def test_write_kaldi_dir(tmp_path):
    utts = ['aaa11111_000', 'aaa11111_001', 'bbb22222_000']
    utt2wav = {u: '/data/{}.wav'.format(u) for u in utts}
    utt2text = {'aaa11111_000': '今日は 良い天気。', 'aaa11111_001': 'こんにちは。', 'bbb22222_000': 'テスト。'}
    utt2spk = {u: u.split('_')[0] for u in utts}
    instruct = 'You are a helpful assistant.<|endofprompt|>'
    prepare.write_kaldi_dir(str(tmp_path), utts, utt2wav, utt2text, utt2spk, instruct)
    text_lines = open(tmp_path / 'text', encoding='utf-8').read().splitlines()
    # text with inner spaces must round-trip via maxsplit=1
    assert text_lines[0].split(maxsplit=1) == ['aaa11111_000', '今日は 良い天気。']
    spk2utt = dict(l.split(maxsplit=1) for l in open(tmp_path / 'spk2utt', encoding='utf-8').read().splitlines())
    assert spk2utt['aaa11111'] == 'aaa11111_000 aaa11111_001'
    instruct_lines = open(tmp_path / 'instruct', encoding='utf-8').read().splitlines()
    assert instruct_lines[2].split(maxsplit=1) == ['bbb22222_000', instruct]

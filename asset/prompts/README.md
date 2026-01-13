# English Prompt Audio Files

Generated using OpenAI TTS API for CosyVoice voice cloning testing.

## File Naming Convention

`en_{gender}_{voice}_{content}.wav`

- `gender`: female / male
- `voice`: nova, shimmer, alloy (female) / echo, fable, onyx (male)
- `content`: greeting, story, technical

## Text Content

### greeting
```
Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions.
```

### story
```
Once upon a time, in a small village by the sea, there lived a young fisherman who dreamed of exploring the world.
```

### technical
```
The new software update includes several performance improvements and bug fixes. Please restart your device after installation.
```

## File List

### Female Voices

| File | Voice | Text |
|------|-------|------|
| `en_female_nova_greeting.wav` | Nova | greeting |
| `en_female_nova_story.wav` | Nova | story |
| `en_female_nova_technical.wav` | Nova | technical |
| `en_female_shimmer_greeting.wav` | Shimmer | greeting |
| `en_female_shimmer_story.wav` | Shimmer | story |
| `en_female_shimmer_technical.wav` | Shimmer | technical |
| `en_female_alloy_greeting.wav` | Alloy | greeting |
| `en_female_alloy_story.wav` | Alloy | story |
| `en_female_alloy_technical.wav` | Alloy | technical |

### Male Voices

| File | Voice | Text |
|------|-------|------|
| `en_male_echo_greeting.wav` | Echo | greeting |
| `en_male_echo_story.wav` | Echo | story |
| `en_male_echo_technical.wav` | Echo | technical |
| `en_male_fable_greeting.wav` | Fable (British) | greeting |
| `en_male_fable_story.wav` | Fable | story |
| `en_male_fable_technical.wav` | Fable | technical |
| `en_male_onyx_greeting.wav` | Onyx (Deep) | greeting |
| `en_male_onyx_story.wav` | Onyx | story |
| `en_male_onyx_technical.wav` | Onyx | technical |

## Usage with CosyVoice ONNX

### Cross-lingual Mode (recommended for different language output)
```bash
python scripts/onnx_inference_pure.py \
    --text "<|en|>Your text to synthesize here." \
    --prompt_wav asset/prompts/en_female_nova_greeting.wav \
    --output output.wav
```

### Zero-shot Mode (same language, needs prompt text)
For zero-shot mode, you need to provide the corresponding prompt text:
```python
# Example: Using nova_greeting
prompt_wav = "asset/prompts/en_female_nova_greeting.wav"
prompt_text = "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions."
```

## Voice Characteristics

| Voice | Gender | Description |
|-------|--------|-------------|
| Nova | Female | Clear, professional |
| Shimmer | Female | Warm, friendly |
| Alloy | Female | Neutral, versatile |
| Echo | Male | Clear, standard |
| Fable | Male | British accent |
| Onyx | Male | Deep, authoritative |

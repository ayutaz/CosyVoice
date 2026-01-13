#!/usr/bin/env python3
"""
Generate English test prompt audio files using OpenAI TTS API.
"""

import os
from pathlib import Path
from openai import OpenAI

# OpenAI voices:
# - alloy: neutral
# - echo: male
# - fable: male (British)
# - onyx: male (deep)
# - nova: female
# - shimmer: female

VOICES = {
    # Female voices
    "nova": "female",
    "shimmer": "female",
    "alloy": "female",  # neutral but sounds more female
    # Male voices
    "echo": "male",
    "fable": "male",
    "onyx": "male",
}

TEXTS = {
    "greeting": "Hello, my name is Sarah. I'm excited to help you with your project today. Let me know if you have any questions.",
    "story": "Once upon a time, in a small village by the sea, there lived a young fisherman who dreamed of exploring the world.",
    "technical": "The new software update includes several performance improvements and bug fixes. Please restart your device after installation.",
}

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAI(api_key=api_key)

    output_dir = Path("asset/prompts")
    output_dir.mkdir(exist_ok=True)

    for voice, gender in VOICES.items():
        for text_name, text in TEXTS.items():
            output_file = output_dir / f"en_{gender}_{voice}_{text_name}.wav"

            print(f"Generating: {output_file.name}")

            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="wav"
            )

            response.stream_to_file(str(output_file))
            print(f"  Saved: {output_file}")

    print(f"\nGenerated {len(VOICES) * len(TEXTS)} audio files in {output_dir}")

if __name__ == "__main__":
    main()

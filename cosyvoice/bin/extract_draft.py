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

"""Extract draft model weights from a CosyVoice3 target model.

Usage:
    python cosyvoice/bin/extract_draft.py \
        --target_model pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
        --output_path pretrained_models/Fun-CosyVoice3-0.5B/llm_draft.pt
"""

import argparse
import torch
from cosyvoice.llm.speculative_decoding import DRAFT_LAYER_INDICES


def extract_draft_weights(target_sd):
    """Extract and remap weights from 24-layer target to 8-layer draft.

    Layer mapping: target [0, 1, 18, 19, 20, 21, 22, 23] -> draft [0..7]
    Also copies: embed_tokens, norm, llm_decoder, speech_embedding.
    """
    draft_sd = {}
    layer_map = {src: dst for dst, src in enumerate(DRAFT_LAYER_INDICES)}

    for key, value in target_sd.items():
        if 'llm.model.model.layers.' in key:
            parts = key.split('.')
            layer_idx_pos = parts.index('layers') + 1
            src_layer = int(parts[layer_idx_pos])
            if src_layer in layer_map:
                parts[layer_idx_pos] = str(layer_map[src_layer])
                new_key = '.'.join(parts)
                draft_sd[new_key] = value
        elif 'llm.model.' in key:
            draft_sd[key] = value
        elif key.startswith('llm_decoder.') or key.startswith('speech_embedding.'):
            draft_sd[key] = value

    return draft_sd


def main():
    parser = argparse.ArgumentParser(description='Extract draft model weights from CosyVoice3 target model')
    parser.add_argument('--target_model', required=True, help='Path to target llm.pt')
    parser.add_argument('--output_path', required=True, help='Output path for llm_draft.pt')
    args = parser.parse_args()

    print('Loading target model from {}...'.format(args.target_model))
    target_sd = torch.load(args.target_model, map_location='cpu', weights_only=True)

    print('Extracting draft weights...')
    print('  Layer mapping: target {} -> draft [0..{}]'.format(DRAFT_LAYER_INDICES, len(DRAFT_LAYER_INDICES) - 1))
    draft_sd = extract_draft_weights(target_sd)

    target_layer_keys = [k for k in target_sd if 'llm.model.model.layers.' in k]
    draft_layer_keys = [k for k in draft_sd if 'llm.model.model.layers.' in k]
    print('  Target layer params: {}'.format(len(target_layer_keys)))
    print('  Draft layer params: {}'.format(len(draft_layer_keys)))
    print('  Total draft params: {}'.format(len(draft_sd)))

    print('Saving draft model to {}...'.format(args.output_path))
    torch.save(draft_sd, args.output_path)
    print('Done.')


if __name__ == '__main__':
    main()

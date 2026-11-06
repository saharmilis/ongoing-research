"""Small closed-set video-action POC. No motion scores or filenames enter the prompt.

Use separate environments for the two documented Transformers versions.
--prepare-only validates clips and emits sampling metadata without loading weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np

LABEL_VERSION = 'construction-actions-poc-1'
MODELS = {
    'd3': ('vllm-sr/d3-nano', '8198c9f763b046605b2f4827b4999971fa2c92c4'),
    'lfm': ('LiquidAI/LFM3-VL-2B', '71d8e745d97c224984e0f20412f532a5ce9dfd3e'),
}
CRITERIA = {
    'standing_still': 'The vehicle stays in place and its working attachment performs no meaningful operation. Ignore small vibration, wind, shadows or foliage motion.',
    'driving': 'The vehicle itself travels across the site; consistent changing viewpoint or background parallax shows translation. Boom or bucket movement alone is not driving.',
    'digging': 'The bucket or tool visibly engages soil/material to excavate, scrape or scoop. Mere movement of an empty attachment is not enough.',
    'loading_or_dumping': 'The attachment visibly transfers or releases material into a truck, container or pile.',
    'tool_repositioning': 'The vehicle stays broadly in place while the boom, arm or bucket is raised, lowered, swung or opened, without clear excavation or material transfer.',
    'other': 'The action is unclear, outside these categories, or mixed with no single dominant action.',
}
STATE = (
    'A chronological silent POV video from a camera mounted on construction equipment. '
    'Classify only the observable action over the clip. Do not infer activity merely '
    'from the equipment being an excavator. Text overlays and surrounding foliage '
    'are not evidence of work. If evidence is insufficient, choose other.'
)
INSTRUCTIONS = 'Which option best describes the dominant observable activity over this clip?'
DEFAULT_VIDEOS = ('standing-1.mp4', 'standing-2.mp4', 'digging-1.mp4')
LABEL_SETS = {
    'basic': ('standing_still', 'driving', 'digging', 'other'),
    'expanded': tuple(CRITERIA),
}


def sample_video(path):
    """Match d3's documented sampling: 2Hz, <=32 evenly spread frames, RGB."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total < 4:
            raise ValueError('Need a valid frame rate and at least four source frames')
        count = min(max(int(total / fps * 2), 4), 32, total)
        indices = np.linspace(0, total - 1, count).round().astype(int).tolist()
        wanted = set(indices)
        frames = []
        index = 0
        height = width = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                height, width = frame.shape[:2]
                # Match the native 512x384 resized baseline, not a thumbnail grid.
                scale = min(1.0, 512 / max(height, width))
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                       interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        if index != total or len(frames) != count:
            raise ValueError('Decoded frames do not match container metadata')
    finally:
        cap.release()
    if len(frames) % 2:
        frames.append(frames[-1])
        indices.append(indices[-1])
    metadata = {'fps': fps, 'total_num_frames': total, 'duration': total / fps,
                'width': frames[0].shape[1], 'height': frames[0].shape[0],
                'frames_indices': indices}
    manifest = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'duration_seconds': total / fps, 'source_fps': fps,
                'source_dimensions': [width, height], 'sampled_frame_count': len(frames),
                'sample_times_seconds': [i / fps for i in indices],
                'prepared_frame_dimensions': [metadata['width'], metadata['height']]}
    return np.stack(frames), metadata, manifest


def rank_scores(scores, criteria=CRITERIA):
    if set(scores) != set(criteria):
        raise ValueError('Model did not return every configured option')
    scores = {key: float(value) for key, value in scores.items()}
    if not all(math.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError('Invalid option scores')
    if not math.isclose(sum(scores.values()), 1, abs_tol=1e-4):
        raise ValueError('Option scores do not sum to one')
    ranked = sorted(scores, key=scores.get, reverse=True)
    margin = scores[ranked[0]] - scores[ranked[1]]
    return {'label': ranked[0], 'option_scores': scores,
            'top_score': scores[ranked[0]], 'top_two_margin': margin,
            'review_suggested': margin < .15 or ranked[0] == 'other',
            'review_policy': 'Margin <0.15 is an unvalidated review flag, not a relabeling rule.'}


class D3:
    def __init__(self, device, cache_dir, criteria):
        from transformers import AutoModel
        model, revision = MODELS['d3']
        self.model = AutoModel.from_pretrained(model, revision=revision,
            trust_remote_code=True, device=device, cache_dir=cache_dir, batch_size=1)
        self.criteria = criteria

    def classify(self, path, frames, metadata):
        # Native video API supplies its own timestamps and published preprocessing.
        raw = self.model.system_one(state=STATE, videos=[str(path)], questions={
            'activity': {'type': 'choice', 'instructions': INSTRUCTIONS, 'criteria': self.criteria}})
        answer = raw['answers']['activity']
        if 'probabilities' not in answer:
            raise RuntimeError('Decision API returned an unsupported/error answer, not a label')
        return rank_scores(answer['probabilities'], self.criteria), raw


class LFM:
    """Adapt the generative VLM to single-token choices; generate no prose."""
    def __init__(self, device, cache_dir, criteria):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model, revision = MODELS['lfm']
        self.torch = torch
        self.device = device
        self.criteria = criteria
        self.processor = AutoProcessor.from_pretrained(model, revision=revision,
            trust_remote_code=True, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(model, revision=revision,
            trust_remote_code=True, cache_dir=cache_dir, dtype=torch.bfloat16,
            device_map={'': device}, attn_implementation='sdpa').eval()
        self.keys = list(criteria)
        self.letters = list('ABCDEF')[:len(criteria)]
        self.tokens = [self.processor.tokenizer.encode(letter, add_special_tokens=False)
                       for letter in self.letters]
        if any(len(ids) != 1 for ids in self.tokens):
            raise ValueError('Letter-choice POC requires one-token option codes')

    def classify(self, path, frames, metadata):
        # No filename, expected label, or motion score appears in the prompt.
        options = '\n'.join(f'{letter}. {key}: {self.criteria[key]}'
                             for letter, key in zip(self.letters, self.keys))
        prompt = STATE + '\n' + INSTRUCTIONS + '\n' + options + f'\nAnswer with exactly one letter A–{self.letters[-1]}.'
        messages = [{'role': 'user', 'content': [
            {'type': 'video'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], videos=[frames], video_metadata=[metadata],
            do_sample_frames=False, return_tensors='pt')
        inputs = {key: value.to(self.device) if hasattr(value, 'to') else value
                  for key, value in inputs.items() if key != 'video_metadata'}
        with self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
            ids = self.torch.tensor([v[0] for v in self.tokens], device=logits.device)
            selected = logits[ids].float()
            probabilities = self.torch.softmax(selected, dim=0).cpu().tolist()
            # Mass outside allowed letters helps reveal whether the constrained
            # answer format itself was natural to the model.
            mass = float(self.torch.exp(self.torch.logsumexp(selected, dim=0)
                         - self.torch.logsumexp(logits.float(), dim=0)).cpu())
        answer = rank_scores(dict(zip(self.keys, probabilities)), self.criteria)
        return answer, {'method': 'single_forward_next_letter_scoring',
            'allowed_letter_probability_mass': mass,
            'option_token_ids': [v[0] for v in self.tokens],
            'input_tokens': int(inputs['input_ids'].shape[1]),
            'generated_tokens': 0, 'thinking_enabled': False}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', nargs='+', default=list(DEFAULT_VIDEOS))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Print the full local exception traceback')
    parser.add_argument('--label-set', choices=LABEL_SETS, default='basic')
    parser.add_argument('--reverse-options', action='store_true', help='Check sensitivity to option order')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    criteria = {key: CRITERIA[key] for key in LABEL_SETS[args.label_set]}
    if args.reverse_options:
        criteria = dict(reversed(list(criteria.items())))
    report = {'poc_version': '1.0', 'label_version': LABEL_VERSION,
        'model_id': MODELS[args.model][0], 'model_revision': MODELS[args.model][1],
        'device': args.device, 'python': platform.python_version(),
        'label_set': args.label_set, 'criteria': criteria, 'state': STATE, 'instructions': INSTRUCTIONS,
        'order_reversed': args.reverse_options, 'execution_status': 'preparing',
        'score_note': 'Option scores are conditional model outputs, not calibrated accuracy probabilities.',
        'results': []}
    paths = [(args.input_dir / name).resolve() for name in args.videos]
    if args.output.resolve() in paths:
        parser.error('Output JSON must not overwrite an input video')
    prepared = []
    for name in args.videos:
        path = args.input_dir / name
        frames, metadata, manifest = sample_video(path)
        prepared.append((path, frames, metadata))
        report['results'].append({'file': path.name, 'input': manifest,
                                   'status': 'prepared_only' if args.prepare_only else 'not_run', 'label': None})
    if args.prepare_only:
        report['execution_status'] = 'prepared_only_no_model_inference'
        save(args.output, report)
        print(f'Prepared {len(prepared)} videos. No model predictions were made.')
        return 0
    stage = 'runtime_setup'
    active_row = None
    try:
        import torch
        import transformers
        required_version = {'d3': '5.17.0', 'lfm': '5.21.0'}[args.model]
        if transformers.__version__.split('+')[0] != required_version:
            raise RuntimeError(f'Use transformers=={required_version} for {args.model}; install its requirements in a separate environment')
        torch.manual_seed(0)
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if device.startswith('cuda') and not torch.cuda.is_bf16_supported():
            raise RuntimeError('A BF16-capable CUDA GPU is required; use --device cpu otherwise')
        report['device'] = device
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report['torch_version'] = torch.__version__
        report['transformers_version'] = transformers.__version__
        stage = 'model_loading'
        started = time.perf_counter()
        engine = (D3 if args.model == 'd3' else LFM)(device, args.cache_dir, criteria)
        report['model_load_seconds'] = time.perf_counter() - started
        stage = 'video_inference'
        for row, (path, frames, metadata) in zip(report['results'], prepared):
            active_row = row
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            answer, raw = engine.classify(path, frames, metadata)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            row.update(answer)
            row.update(status='completed', inference_seconds=time.perf_counter()-started,
                       raw_model_response=raw)
            report['execution_status'] = 'partial'
            save(args.output, report)
            print(json.dumps({'file': row['file'], 'label': row['label'],
                              'scores': row['option_scores']}), flush=True)
        report['execution_status'] = 'completed'
        save(args.output, report)
        return 0
    except Exception as error:
        # Keep honest partial results, and never replace model failures with Other.
        # Do not persist signed URLs, credentials, or arbitrary exception strings.
        report['execution_status'] = 'failed_or_blocked'
        if active_row is not None:
            active_row['status'] = 'failed'
        if args.debug:
            import traceback
            traceback.print_exc()
        report['error'] = {'type': type(error).__name__, 'stage': stage,
                          'message': 'Model setup/inference failed; inspect local stderr and environment.'}
        save(args.output, report)
        print(f'{type(error).__name__} during {stage}: model setup/inference failed. Use --debug for the local traceback. No substitute predictions were created.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

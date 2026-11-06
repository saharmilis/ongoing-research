"""Construction POV motion graph: CPU detection, fixed calibration, temporal smoothing.

Run: python motion_graph.py video.mp4 --output result.json --plot graph.png
Only NumPy/OpenCV are required for analysis. Matplotlib is optional for exports.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from smoothing import GENTLE, STRONGER, smooth_scores

ALGORITHM_VERSION = 'contech-motion-1.0.0'
SCHEMA_VERSION = '1.0'
KERNELS = {'none': (1,), 'gentle': GENTLE, 'stronger': STRONGER}


@dataclass(frozen=True)
class Config:
    width: int = 320
    sample_fps: int = 5
    lag_samples: int = 3
    blur_size: int = 5
    pixel_threshold: float = 12.0
    min_component_fraction: float = 0.001
    area_deadband: float = 0.01
    calibration_beta: float = 0.15
    smoothing: str = 'stronger'

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.width, self.sample_fps, self.lag_samples, self.blur_size)):
            raise ValueError('Dimensions and sampling parameters must be positive integers')
        if self.width < 32 or self.blur_size % 2 != 1:
            raise ValueError('Width must be >=32; blur_size must be odd')
        for name in ('pixel_threshold', 'min_component_fraction', 'area_deadband', 'calibration_beta'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 0 < self.pixel_threshold < 255:
            raise ValueError('pixel_threshold must be between 0 and 255')
        if not 0 <= self.min_component_fraction < 1 or not 0 <= self.area_deadband < 1:
            raise ValueError('Area fractions must be in [0,1)')
        if self.calibration_beta <= 0 or self.smoothing not in KERNELS:
            raise ValueError('Invalid calibration beta or smoothing mode')


def prepare_frame(frame, config):
    height, width = frame.shape[:2]
    target = min(width, config.width)
    resized = cv2.resize(frame, (target, max(1, round(height * target / width))),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (config.blur_size, config.blur_size), 0)


def measure_change(previous, current, config):
    if previous.shape != current.shape:
        raise ValueError('Video dimensions changed during decoding')
    difference = current.astype(np.float32) - previous.astype(np.float32)
    offset = float(np.median(difference))
    difference -= offset
    mask = (np.abs(difference) > config.pixel_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(1, math.ceil(mask.size * config.min_component_fraction))
    keep[0] = False
    area = float(keep[labels].mean())
    detector = float(np.clip((area - config.area_deadband) / (1 - config.area_deadband), 0, 1))
    calibrated = (1 + config.calibration_beta) * detector / (detector + config.calibration_beta)
    return {'changed_area_fraction': area, 'detector_score': detector,
            'calibrated_score': float(calibrated), 'brightness_offset': offset}


def _sample_frames(path, config, metadata, stop_seconds=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps < config.sample_fps:
        cap.release()
        raise ValueError('Source must report a valid frame rate >= sample_fps')
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata.update(fps_reported=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), decoded_frames=0,
                    reached_eof=False)
    origin = None
    previous_pts = -math.inf
    tick = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                metadata['reached_eof'] = True
                if reported_frames > 0 and metadata['decoded_frames'] != reported_frames:
                    raise ValueError('Decoded frame count differs from container metadata; validate input')
                break
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(pts) or pts <= previous_pts:
                raise ValueError('Missing or non-monotonic decoded timestamps')
            interval = pts - previous_pts if math.isfinite(previous_pts) else 1 / fps
            previous_pts = pts
            if origin is None:
                origin = pts
                metadata['first_decoded_timestamp_seconds'] = pts
            relative = pts - origin
            metadata['decoded_frames'] += 1
            metadata['decoded_through_seconds'] = relative + interval
            if stop_seconds is not None and relative >= stop_seconds - 1e-6:
                break
            target = tick / config.sample_fps
            if relative + 1e-6 >= target:
                if relative - target > 1 / config.sample_fps + 1e-6:
                    raise ValueError('Timestamp gap exceeds sampling interval; split at recording gaps')
                yield target, relative, prepare_frame(frame, config)
                tick += 1
    finally:
        cap.release()


def _window_bound(value, name):
    if value is not None and (not math.isfinite(value) or value < 0 or not float(value).is_integer()):
        raise ValueError(f'{name} must be a finite nonnegative whole number of seconds')


def _nullable(value):
    return float(value) if np.isfinite(value) else None


def analyze_video(path, config=None, *, start_seconds=0, end_seconds=None, include_samples=False):
    """Analyze a validated local video, retaining neighboring footage for smoothing.

    Bounds refer to the input file, relative to its first decoded frame. Requested
    bounds must be whole seconds. Output timestamps are relative to start_seconds.
    Default: selected smooth calibration plus five-second temporal smoothing.
    """
    config = config or Config()
    _window_bound(start_seconds, 'start_seconds')
    _window_bound(end_seconds, 'end_seconds')
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError('end_seconds must be greater than start_seconds')
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Video is not a local file: {path}')
    started = time.perf_counter()
    cv2.setNumThreads(1)
    radius = len(KERNELS[config.smoothing]) // 2
    stop = None if end_seconds is None else end_seconds + radius
    metadata = {}
    history = deque(maxlen=config.lag_samples)
    samples = []
    for tick, pts, frame in _sample_frames(path, config, metadata, stop):
        if len(history) == config.lag_samples:
            prev_pts, previous = history[0]
            row = measure_change(previous, frame, config)
            row.update(input_time_seconds=round(tick, 6),
                       decoded_time_seconds=round(pts, 6),
                       comparison_interval_seconds=round(pts - prev_pts, 6))
            samples.append(row)
        history.append((pts, frame))
    if not samples:
        raise ValueError('Insufficient valid frames: at least 0.6s of comparison context is needed')
    available_end = round(metadata['decoded_through_seconds'], 9)
    if end_seconds is not None and metadata['reached_eof'] and end_seconds > available_end + 1e-6:
        raise ValueError('Requested end_seconds is beyond decoded video duration')
    output_end = float(end_seconds) if end_seconds is not None else available_end
    if start_seconds >= output_end:
        raise ValueError('start_seconds is outside the decoded video')

    # Bucket by input clock, including context. Apply calibration before averaging.
    buckets = {}
    for row in samples:
        buckets.setdefault(math.floor(row['input_time_seconds']), []).append(row)
    count = math.ceil(min(available_end, stop) if stop is not None else available_end)
    scores = np.full(count, np.nan)
    for index, members in buckets.items():
        if index < count:
            scores[index] = np.mean([m['calibrated_score'] for m in members])
    smoothed = smooth_scores(scores, KERNELS[config.smoothing])
    rows = []
    first = int(start_seconds)
    for index in range(first, math.ceil(output_end)):
        members = buckets.get(index, [])
        end = min(index + 1, output_end)
        expected = sum(index <= k / config.sample_fps < end - 1e-9
                       for k in range(index * config.sample_fps, (index + 1) * config.sample_fps))
        padded = index - radius < 0 or index + radius >= len(scores)
        if not padded:
            padded = not np.isfinite(scores[index-radius:index+radius+1]).all()
        row = {'start_seconds': round(index - start_seconds, 9),
               'end_seconds': round(end - start_seconds, 9),
               'input_start_seconds': index,
               'score': _nullable(smoothed[index]),
               'calibrated_score': _nullable(scores[index]),
               'detector_score': float(np.mean([m['detector_score'] for m in members])) if members else None,
               'changed_area_fraction': float(np.mean([m['changed_area_fraction'] for m in members])) if members else None,
               'sample_count': len(members), 'expected_sample_count': expected,
               'coverage_fraction': len(members) / expected if expected else 0.0,
               'smoothing_edge_padded': bool(padded)}
        rows.append(row)
    target_samples = [s for s in samples if start_seconds <= s['input_time_seconds'] < output_end]
    if not target_samples:
        raise ValueError('No valid comparison measurements in the requested window')
    valid = [r['score'] for r in rows if r['score'] is not None]
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    result = {'schema_version': SCHEMA_VERSION, 'algorithm_version': ALGORITHM_VERSION,
              'source': {'name': path.name, 'sha256': digest.hexdigest(), **metadata},
              'config': {**asdict(config), 'smoothing_weights': list(KERNELS[config.smoothing])},
              'window': {'input_start_seconds': start_seconds, 'input_end_seconds': output_end,
                         'duration_seconds': round(output_end - start_seconds, 9),
                         'unscored_prefix_seconds': max(0, target_samples[0]['input_time_seconds'] - start_seconds)},
              'score': float(np.mean(valid)), 'peak_score': float(max(valid)),
              'calibrated_sample_mean': float(np.mean([s['calibrated_score'] for s in target_samples])),
              'sample_count': len(target_samples), 'scores': [r['score'] for r in rows],
              'seconds': rows, 'elapsed_seconds': time.perf_counter() - started}
    if include_samples:
        result['samples'] = [{**s, 'time_seconds': round(s['input_time_seconds'] - start_seconds, 6)}
                             for s in target_samples]
    return result


def export_plot(result, png_path=None, html_path=None):
    """Optional Matplotlib exports; HTML is self-contained and embeds exact JSON."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Plot export requires: pip install -r requirements-plot.txt') from error
    rows = result['seconds']
    x = [(r['start_seconds'] + r['end_seconds']) / 2 for r in rows]
    y = [r['score'] if r['score'] is not None else np.nan for r in rows]
    raw = [r['calibrated_score'] if r['calibrated_score'] is not None else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.2, top=.85)
    ax.plot(x, raw, color='#a5acb5', lw=1, label='Calibrated, unsmoothed')
    ax.plot(x, y, color='#d17715', lw=1.8, label=f"Final: {result['config']['smoothing']}")
    ax.fill_between(x, y, color='#d17715', alpha=.12)
    ax.set(xlim=(0, result['window']['duration_seconds']), ylim=(-.02, 1.02),
           xlabel='Time relative to requested window (seconds)', ylabel='Motion score (0–1)')
    prefix = result['window']['unscored_prefix_seconds']
    if prefix:
        ax.axvspan(0, prefix, color='#64748b', alpha=.15)
    ax.set_title(f"{result['source']['name']} — motion graph", loc='left', fontsize=17)
    ax.grid(alpha=.15)
    ax.legend(loc='upper right')
    fig.text(.07, .04, f"{ALGORITHM_VERSION} | Mean {result['score']:.4f} | Peak {result['peak_score']:.4f} | Fixed calibration; no per-video normalization", fontsize=10)
    if png_path:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150, facecolor='white')
    if html_path:
        svg = io.StringIO()
        fig.savefig(svg, format='svg', facecolor='white')
        content = svg.getvalue()
        content = content[content.index('<svg'):]
        payload = json.dumps(result, allow_nan=False).replace('</', '<\\/')
        page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion graph</title><style>body{font:16px system-ui;margin:24px;color:#172435}main{max-width:1450px;margin:auto}svg{width:100%;height:auto}button{padding:12px}p{line-height:1.6}</style><main><h1>Motion graph</h1>''' + content + '''<p>Gray: calibrated scores before smoothing. Orange: final scores. Shaded initial coverage lacks comparison context. Source timestamps and per-second coverage are included in the downloadable results. This page works offline.</p><button id="download">Download results JSON</button></main><script id="data" type="application/json">''' + payload + '''</script><script>const r=JSON.parse(document.getElementById('data').textContent);document.getElementById('download').addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='motion-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});</script></html>'''
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page, encoding='utf-8')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, help='JSON path; otherwise print JSON to stdout')
    parser.add_argument('--plot', type=Path, help='Optional PNG output')
    parser.add_argument('--html', type=Path, help='Optional self-contained HTML output')
    parser.add_argument('--smoothing', choices=KERNELS, default='stronger')
    parser.add_argument('--start-seconds', type=float, default=0)
    parser.add_argument('--end-seconds', type=float)
    parser.add_argument('--include-samples', action='store_true')
    args = parser.parse_args()
    try:
        # Never overwrite a recording with a result or collide export paths.
        outputs = [p for p in (args.output, args.plot, args.html) if p is not None]
        resolved = [p.resolve() for p in outputs]
        if args.video.resolve() in resolved or len(set(resolved)) != len(resolved):
            raise ValueError('Input and output paths must be distinct')
        result = analyze_video(args.video, Config(smoothing=args.smoothing),
                               start_seconds=args.start_seconds, end_seconds=args.end_seconds,
                               include_samples=args.include_samples)
        if args.plot or args.html:
            export_plot(result, args.plot, args.html)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        else:
            print(text)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'motion_graph: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

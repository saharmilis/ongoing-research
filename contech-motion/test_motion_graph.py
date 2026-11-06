"""Numerical, media, chunk, CLI and real-baseline handoff checks."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from motion_graph import Config, analyze_video, measure_change, prepare_frame
from smoothing import GENTLE, STRONGER, smooth_scores

ROOT = Path(__file__).parent


def write_video(path, frames, fps=25):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
    assert writer.isOpened(), 'Test fixture needs FFV1 codec support'
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def frames():
    result = []
    for i in range(250):
        frame = np.full((240, 320, 3), 50, np.uint8)
        x = (i * 2) % 240
        frame[60:170, x:x+80] = 180
        result.append(frame)
    return result


def test_math_noise_and_small_regions():
    c = Config()
    a = np.full((240, 320, 3), 80, np.uint8)
    b = a.copy()
    b[80:88, 90:98] = 200
    for image in [a.copy(), a + np.uint8(15), b]:
        assert measure_change(prepare_frame(a, c), prepare_frame(image, c), c)['calibrated_score'] == 0
    b = a.copy()
    b[60:180, 80:240] = 200
    r = measure_change(prepare_frame(a, c), prepare_frame(b, c), c)
    assert .2 < r['changed_area_fraction'] < .3
    assert .6 < r['calibrated_score'] < .8


@pytest.mark.parametrize('mode', ['none', 'gentle', 'stronger'])
def test_chunk_context_matches_continuous(tmp_path, frames, mode):
    full_path = tmp_path / 'full.avi'
    write_video(full_path, frames)
    config = Config(smoothing=mode)
    whole = analyze_video(full_path, config)
    left = analyze_video(full_path, config, end_seconds=5)
    # Right input starts at source second 2; target second 5 has 3s prefix.
    right_path = tmp_path / 'right.avi'
    write_video(right_path, frames[50:])
    right = analyze_video(right_path, config, start_seconds=3)
    np.testing.assert_allclose(left['scores'] + right['scores'], whole['scores'], atol=0, rtol=0)
    assert right['window']['unscored_prefix_seconds'] == 0
    assert right['seconds'][0]['smoothing_edge_padded'] is False


def test_smoothing_preserves_constants_and_gaps():
    for weights in [GENTLE, STRONGER]:
        np.testing.assert_allclose(smooth_scores([.4] * 8, weights), .4)
        values = smooth_scores([0, 0, None, 1, 1], weights)
        assert np.isnan(values[2])
        np.testing.assert_allclose(values[[0, 1, 3, 4]], [0, 0, 1, 1])
    np.testing.assert_allclose(smooth_scores([0, 0, 1, 0, 0]), [0, .25, .5, .25, 0])


def test_short_video_is_unmeasured_not_zero(tmp_path, frames):
    path = tmp_path / 'short.avi'
    write_video(path, frames[:10])
    with pytest.raises(ValueError, match='Insufficient'):
        analyze_video(path)


def test_invalid_input_and_bounds(tmp_path, frames):
    with pytest.raises(ValueError):
        analyze_video(tmp_path / 'missing.mp4')
    path = tmp_path / 'valid.avi'
    write_video(path, frames[:50])
    for start, end in [(.3, None), (float('nan'), None), (0, 5), (0, 0), (3, None)]:
        with pytest.raises(ValueError):
            analyze_video(path, start_seconds=start, end_seconds=end)
    with pytest.raises(ValueError):
        Config(calibration_beta=0)


def test_partial_coverage_and_schema(tmp_path, frames):
    path = tmp_path / 'valid.avi'
    write_video(path, frames[:50])
    r = analyze_video(path, include_samples=True)
    assert len(r['scores']) == 2
    assert r['window']['unscored_prefix_seconds'] == .6
    assert r['seconds'][0]['sample_count'] == 2
    assert r['seconds'][0]['coverage_fraction'] == .4
    assert r['seconds'][1]['sample_count'] == 5
    assert r['source']['reached_eof'] is True
    assert len(r['samples']) == 7
    json.dumps(r, allow_nan=False)


def test_cli_exports_and_input_protection(tmp_path, frames):
    path = tmp_path / 'video.avi'
    write_video(path, frames[:75])
    result, png, html = [tmp_path / name for name in ['result.json', 'plot.png', 'plot.html']]
    subprocess.run([sys.executable, str(ROOT/'motion_graph.py'), str(path),
                    '--output', str(result), '--plot', str(png), '--html', str(html)], check=True)
    r = json.loads(result.read_text())
    assert len(r['scores']) == 3
    assert png.read_bytes().startswith(b'\x89PNG')
    assert '<svg' in html.read_text() and 'application/json' in html.read_text()
    protected = subprocess.run([sys.executable, str(ROOT/'motion_graph.py'), str(path),
                                '--output', str(path)], capture_output=True)
    assert protected.returncode == 1
    assert b'paths must be distinct' in protected.stderr


@pytest.mark.parametrize('name, mean, peak', [
    ('standing-1.mp4', 0, 0),
    ('standing-2.mp4', 0, 0),
    ('digging-1.mp4', .6572938352158653, .8372587398239068),
])
def test_original_real_baselines(name, mean, peak):
    r = analyze_video(ROOT/name)
    assert len(r['seconds']) == 15
    assert r['score'] == pytest.approx(mean, abs=1e-12)
    assert r['peak_score'] == pytest.approx(peak, abs=1e-12)


def test_five_minute_selected_baseline():
    # Genuine future context after 300s is included automatically.
    r = analyze_video(ROOT/'5min.mp4', end_seconds=300)
    assert len(r['scores']) == 300
    assert r['sample_count'] == 1497
    assert r['scores'][-1] == pytest.approx(0.4902213037734363, abs=1e-12)
    assert all(v is not None and 0 <= v <= 1 for v in r['scores'])
    assert r['source']['reached_eof'] is False
    assert r['seconds'][-1]['smoothing_edge_padded'] is False
    # Recorded from the previously approved smooth-calibration + stronger plot.
    assert r['score'] == pytest.approx(0.5309664266038264, abs=1e-12)


def test_low_frame_rate_is_explicitly_unsupported(tmp_path, frames):
    path = tmp_path / 'slow.avi'
    write_video(path, frames[:12], fps=4)
    with pytest.raises(ValueError, match='frame rate'):
        analyze_video(path)


def test_invalid_smoothing_and_empty_runs():
    assert len(smooth_scores([])) == 0
    for weights in [[], [1, 1], [-1, 2, 1], [0, 0, 0]]:
        with pytest.raises(ValueError):
            smooth_scores([.1, .2, .3], weights)
    with pytest.raises(ValueError):
        smooth_scores([1.1])

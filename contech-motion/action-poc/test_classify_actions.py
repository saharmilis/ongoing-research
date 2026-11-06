import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from classify_actions import CRITERIA, LABEL_SETS, MODELS, STATE, rank_scores, sample_video

ROOT = Path(__file__).parent
VIDEOS = ROOT.parent


def test_actual_inputs_have_temporal_frames():
    for name in ['standing-1.mp4', 'standing-2.mp4', 'digging-1.mp4']:
        frames, metadata, manifest = sample_video(VIDEOS/name)
        assert frames.shape == (30, 384, 512, 3)
        assert frames.dtype == np.uint8
        assert metadata['fps'] == 25
        assert metadata['total_num_frames'] == 375
        times = manifest['sample_times_seconds']
        assert times[0] == 0 and times[-1] == 14.96
        assert all(a < b for a,b in zip(times,times[1:]))
        assert len(manifest['source_sha256']) == 64


def test_no_filenames_or_expected_labels_in_state():
    assert not any(name in STATE for name in ['standing-1', 'standing-2', 'digging-1'])
    assert set(LABEL_SETS['basic']) == {'standing_still', 'driving', 'digging', 'other'}
    assert all(len(revision) == 40 for _,revision in MODELS.values())


def test_ranking_preserves_other_and_uncertainty():
    scores = dict.fromkeys(CRITERIA, .04)
    scores['other'] = .8
    r = rank_scores(scores)
    assert r['label'] == 'other' and r['review_suggested']
    scores = dict.fromkeys(CRITERIA, 1/6)
    r = rank_scores(scores)
    assert r['review_suggested'] and r['top_two_margin'] == 0
    with pytest.raises(ValueError):
        rank_scores({'digging': float('nan')})
    basic = {k: .25 for k in LABEL_SETS['basic']}
    assert rank_scores(basic, basic)['label'] in basic


def test_prepare_only_reports_no_predictions(tmp_path):
    output=tmp_path/'prepared.json'
    subprocess.run([sys.executable,str(ROOT/'classify_actions.py'), '--model','d3',
                    '--prepare-only','--output',str(output)],check=True)
    r=json.loads(output.read_text())
    assert r['execution_status']=='prepared_only_no_model_inference'
    assert len(r['results'])==3
    assert all(row['label'] is None for row in r['results'])
    assert len(r['criteria'])==4


def test_output_cannot_overwrite_source():
    source=VIDEOS/'standing-1.mp4'
    before=source.stat().st_size
    r=subprocess.run([sys.executable,str(ROOT/'classify_actions.py'), '--model','d3',
                      '--prepare-only','--output',str(source)],capture_output=True,text=True)
    assert r.returncode!=0 and 'must not overwrite' in r.stderr
    assert source.stat().st_size==before


def test_summary_does_not_invent_predictions(tmp_path):
    from summarize_results import summarize
    r={'model_id':'example/model','results':[{'file':'clip.mp4','status':'prepared_only','label':None}]}
    p=tmp_path/'result.json';p.write_text(json.dumps(r))
    text=summarize([p])
    assert 'NOT RUN' in text and '| other |' not in text

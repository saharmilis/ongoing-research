"""Summarize actual model result JSONs. Never turn missing inference into Other."""
import argparse
import json
from pathlib import Path


def summarize(paths):
    lines = ['| Model | Video | Status | Label | Top score | Margin | Seconds |',
             '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for path in paths:
        report = json.loads(Path(path).read_text())
        for row in report['results']:
            completed = row.get('status') == 'completed'
            def number(key):
                value = row.get(key)
                return f'{value:.4f}' if completed and value is not None else '—'
            label = row.get('label') if completed else 'NOT RUN'
            lines.append(f"| {report['model_id']} | {row['file']} | {row['status']} | {label} | {number('top_score')} | {number('top_two_margin')} | {number('inference_seconds')} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    args = parser.parse_args()
    print(summarize(args.results))

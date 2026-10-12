"""Read-only Hugging Face paper retrieval; immutable evidence, no GPU work."""
import argparse
import datetime
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def fetch(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={
            'User-Agent': 'drone-rl-lab-methodology/1.0'}), timeout=30) as response:
            return {'url': url, 'status': response.status,
                    'body': response.read(8_000_000).decode('utf-8')}
    except (urllib.error.URLError, TimeoutError) as error:
        return {'url': url, 'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--paper', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.query) > 250:
        parser.error('HF query must be <=250 characters')
    for paper in args.paper:
        if not re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', paper):
            parser.error('Invalid arXiv ID: ' + paper)
    # Refuse overwrite even when the earlier retrieval failed.
    args.output.mkdir(parents=True, exist_ok=False)
    records = [fetch('https://huggingface.co/api/papers/search?' +
                     urllib.parse.urlencode({'q': args.query, 'limit': 8}))]
    for paper in args.paper:
        records.extend([fetch(f'https://huggingface.co/api/papers/{paper}'),
                        fetch(f'https://huggingface.co/papers/{paper}.md')])
    manifest = {'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'query': args.query, 'paper_ids': args.paper, 'requests': []}
    for index, record in enumerate(records):
        body = record.pop('body', None)
        if body is not None:
            name = f'source_{index:02d}.txt'
            (args.output / name).write_text(body, encoding='utf-8')
            record['file'] = name
        manifest['requests'].append(record)
    (args.output / 'retrieval.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

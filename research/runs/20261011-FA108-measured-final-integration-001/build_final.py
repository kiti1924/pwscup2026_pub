"""FA108: copy measured whole target vectors; never predict, score or upload."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research'))
from package_public_attack import run as package


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'candidate').exists(), 'Immutable output already exists'
    plan = json.loads((OUT / 'plan.json').read_text())
    for path, expected in json.loads((OUT / 'input_hashes.json').read_text()).items():
        assert sha((ROOT / path).read_bytes()) == expected, path
    routing = json.loads((ROOT / plan['routing_source']).read_text())
    cache = {}
    header = b'target_k,challenge_row_id,time_hat'
    lines = []
    bindings = []
    for route in routing['AIA']:
        path = route['source_zip']
        if path not in cache:
            raw_zip = (ROOT / path).read_bytes()
            assert sha(raw_zip) == route['source_zip_sha256'], path
            with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
                assert len(z.namelist()) == 3 and set(z.namelist()) == {'F_aia.csv', 'F_mia.csv', 'token.txt'}
                cache[path] = {n: z.read(n) for n in z.namelist()}
        original = cache[path]['F_aia.csv']
        assert sha(original) == route['source_AIA_member_sha256']
        assert b'\r' not in original and original.endswith(b'\n'), 'Source must be LF'
        source_lines = original.splitlines()
        assert source_lines[0] == header
        k = route['target_k']
        selected = [line for line in source_lines[1:] if int(line.split(b',', 1)[0]) == k]
        assert len(selected) == 326
        values = list(csv.DictReader(io.StringIO((header + b'\n' + b'\n'.join(selected) + b'\n').decode())))
        assert len({v['challenge_row_id'] for v in values}) == 326
        lines.extend(selected)
        bindings.append({**route, 'selected_lines_sha256': sha(b'\n'.join(selected) + b'\n'), 'all326_lines_exact': True})
    assert len(lines) == 7172
    aia = header + b'\n' + b'\n'.join(lines) + b'\n'
    (OUT / 'F_aia.csv').write_bytes(aia)
    mia_route = routing['MIA']
    with zipfile.ZipFile(ROOT / mia_route['source_zip']) as z:
        mia = z.read('F_mia.csv')
    assert sha((ROOT / mia_route['source_zip']).read_bytes()) == mia_route['source_zip_sha256']
    assert sha(mia) == mia_route['source_MIA_member_sha256']
    with zipfile.ZipFile(ROOT / 'research/submissions/four-measured-backup-20261010/candidate/candidate.zip') as z:
        assert mia == z.read('F_mia.csv')
        token = z.read('token.txt')
    assert token == (ROOT / '_private_team14/token.txt').read_bytes()
    for parts in cache.values():
        assert parts['token.txt'] == token
    (OUT / 'F_mia.csv').write_bytes(mia)
    package(OUT)
    with zipfile.ZipFile(OUT / 'candidate/candidate.zip') as z:
        assert len(z.namelist()) == 3
        assert z.read('F_aia.csv') == aia and z.read('F_mia.csv') == mia and z.read('token.txt') == token
        # Independent selection from final member verifies original line text, not just floats.
        final_lines = z.read('F_aia.csv').splitlines()[1:]
        for binding in bindings:
            chosen = [line for line in final_lines if int(line.split(b',', 1)[0]) == binding['target_k']]
            assert sha(b'\n'.join(chosen) + b'\n') == binding['selected_lines_sha256']
    counts = dict(sorted(Counter(str(r['source_submission']) for r in bindings).items()))
    dump(OUT / 'source_binding.json', {'AIA': bindings, 'MIA': mia_route, 'source_counts': counts,
         'original326_lines_exact': True, 'MIA_whole_bytes_exact': True, 'token_backup_bytes_exact': True,
         'three_unique_entries': True, 'no_new_predictions': True, 'expected_total': routing['total_calculated']})
    dump(OUT / 'final_routing.json', routing)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    # Generic packager metadata is specialized without changing its validation or inputs.
    manifest.pop('FA98_screen', None)
    manifest.update({'experiment': 'FA108', 'main_measured': False, 'expected_total': routing['total_calculated'],
                     'expected_AIA_mean': routing['aia_mean'], 'expected_MIA_power': mia_route['measured_power'],
                     'MIA_source_submission': 972302, 'source_counts': counts, 'new_research': False})
    dump(OUT / 'manifest.json', manifest)
    dump(OUT / 'builder_hash.json', {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__).read_bytes())})
    print('FA108 ready', manifest['zip_sha256'], manifest['zip_bytes'], 'expected', routing['total_calculated'], 'sources', counts)


if __name__ == '__main__':
    main()

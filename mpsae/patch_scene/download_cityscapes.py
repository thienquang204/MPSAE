"""Download only the two official fine packages; credentials stay in memory.

CLI reads a JSON object {username, password} from stdin, never command arguments.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

PACKAGES = ('leftImg8bit_trainvaltest.zip', 'gtFine_trainvaltest.zip')
BASE = 'https://www.cityscapes-dataset.com'


def md5(path):
    h = hashlib.md5()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(session, package_id, target, digest):
    from tqdm import tqdm
    target = Path(target)
    partial = target.with_suffix(target.suffix+'.partial')
    if target.exists():
        if md5(target) != digest:
            raise RuntimeError(f'Checksum mismatch: {target}. Move the damaged file before retrying.')
        print(f'[reuse verified download] {target.name}', flush=True)
        return
    if partial.exists() and md5(partial) == digest:
        partial.replace(target)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    url = f'{BASE}/file-handling/?packageID={package_id}'
    response = session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                           stream=True, allow_redirects=False, timeout=(30, 120))
    if response.status_code == 416:
        response.close()
        offset = 0
        response = session.get(url, stream=True, allow_redirects=False, timeout=(30,120))
    with response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f'Cityscapes download failed: HTTP {response.status_code}')
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise RuntimeError('Cityscapes returned a web page instead of a ZIP; check account access')
        if response.status_code == 206:
            if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise RuntimeError('Unexpected resumed byte range')
        else:
            offset = 0  # server ignored Range: replace the partial, never append twice
        length = response.headers.get('Content-Length')
        total = offset+int(length) if length else None
        with partial.open('ab' if offset else 'wb') as handle, tqdm(total=total, initial=offset,
                desc=target.name, unit='B', unit_scale=True) as progress:
            for chunk in response.iter_content(chunk_size=2 << 20):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    if md5(partial) != digest:
        raise RuntimeError(f'Checksum mismatch for {target.name}; download is not accepted')
    partial.replace(target)


def download(destination, username, password):
    import requests
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.get(BASE+'/login', allow_redirects=False, timeout=30).raise_for_status()
        response = session.post(BASE+'/login', data={'username': username, 'password': password,
                                'submit': 'Login'}, allow_redirects=False, timeout=30)
        if response.status_code != 302:
            raise RuntimeError('Cityscapes login failed. Register/activate an account and accept its terms first.')
        response = session.get(BASE+'/downloads/?list', allow_redirects=False, timeout=30)
        response.raise_for_status()
        available = {p['name']: p['packageID'] for p in response.json()}
        for name in PACKAGES:
            if name not in available:
                raise RuntimeError(f'Your Cityscapes account cannot access {name}')
            package_id = available[name]
            response = session.get(BASE+f'/md5-sum/?packageID={package_id}', timeout=30)
            response.raise_for_status()
            digest = response.text.split()[0].lower()
            if not re.fullmatch('[a-f0-9]{32}', digest):
                raise RuntimeError('Invalid official checksum response')
            fetch(session, package_id, root/name, digest)


if __name__ == '__main__':
    credentials = json.loads(sys.stdin.read())
    try:
        download(sys.argv[1], credentials['username'], credentials['password'])
    finally:
        credentials.clear()

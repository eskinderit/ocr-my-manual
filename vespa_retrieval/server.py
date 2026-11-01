"""Manage an optional local llama-server without leaving a GPU process behind."""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response).get('status') == 'ok'
    except (OSError, ValueError, urllib.error.URLError):
        return False


@contextmanager
def local_server(binary: Path, model: Path, projector: Path, log_path: Path, port: int = 8003):
    for path in (binary, model, projector):
        if not path.is_file():
            raise FileNotFoundError(f'Missing local OCR runtime asset: {path}. Supply paths or --endpoint for an existing server.')
    health = f'http://127.0.0.1:{port}/health'
    if server_ready(health):
        raise RuntimeError(f'A server is already running on port {port}; use --endpoint to reuse it explicitly')
    command = [str(binary.resolve()), '-m', str(model.resolve()), '--mmproj', str(projector.resolve()),
               '--host', '127.0.0.1', '--port', str(port), '--alias', 'qwen3vl4b',
               '-ngl', '99', '-c', '8192', '-np', '1', '-fa', 'on', '--jinja']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Native builds may embed their original absolute build directory in RUNPATH.
    # Resolve sibling shared libraries even after the project folder is moved.
    environment['LD_LIBRARY_PATH'] = str(binary.resolve().parent) + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 180
            while not server_ready(health):
                if process.poll() is not None:
                    raise RuntimeError(f'OCR server exited; inspect {log_path}')
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'OCR server startup timed out; inspect {log_path}')
                time.sleep(0.5)
            if process.poll() is not None:
                raise RuntimeError(f'OCR server exited; inspect {log_path}')
            yield f'http://127.0.0.1:{port}/v1/chat/completions'
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
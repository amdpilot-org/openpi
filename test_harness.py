#!/usr/bin/env python3
import json
import pathlib
import re
import subprocess
import sys

cmd = ['/bin/bash', '/workspace/bench_openpi.sh'] if pathlib.Path('/workspace/bench_openpi.sh').exists() else ['/bin/bash', '/opt/openpi/bench_openpi.sh']
log_path = pathlib.Path('/tmp/openpi_harness.log')
metric = None
with log_path.open('w') as log:
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
        m = re.search(r'STAGE0_METRIC\s+throughput_inf_s=([0-9.]+)', line)
        if m:
            metric = float(m.group(1))
    rc = proc.wait(timeout=30)
if rc != 0:
    sys.exit(rc)
if metric is None:
    p = pathlib.Path('/tmp/openpi_result.json')
    if p.exists():
        metric = json.loads(p.read_text()).get('throughput_inf_s')
if metric is None:
    text = log_path.read_text(errors='replace')
    m = re.search(r'SCORE:\s*([0-9.]+)\s*throughput_inf_s', text) or re.search(r'Throughput:\s*([0-9.]+)\s*inf/s', text)
    if m:
        metric = float(m.group(1))
if metric is None:
    print('No throughput metric extracted', file=sys.stderr)
    sys.exit(2)
print(json.dumps({'status': 'pass', 'throughput_inf_s': metric}))

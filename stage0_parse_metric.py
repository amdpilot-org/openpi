import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors='replace')
match = re.search(r'SCORE:\s*([0-9.]+)\s*throughput_inf_s', log) or re.search(r'Throughput:\s*([0-9.]+)\s*inf/s', log)
if not match:
    raise SystemExit('throughput metric not found')
value = float(match.group(1))
out = Path(sys.argv[2])
try:
    data = json.loads(out.read_text()) if out.exists() else {}
except Exception:
    data = {}
data.update({'throughput_inf_s': value, 'metric_name': 'throughput_inf_s', 'metric_direction': 'higher'})
out.write_text(json.dumps(data, indent=2))

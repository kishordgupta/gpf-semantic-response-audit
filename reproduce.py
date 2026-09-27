"""Rebuild the audit, embedding analyses, validation, and dashboard."""
import gzip, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
raw=ROOT/'data/responses.csv'
if not raw.exists():
    with gzip.open(str(raw)+'.gz','rb') as f: raw.write_bytes(f.read())
for name in ['audit_data.py','analyze.py','screen_analysis.py','validate_chunking.py','validate_results.py','build_dashboard.py']:
    p=ROOT/name
    if not p.exists():raise FileNotFoundError(p)
    subprocess.run([sys.executable,str(p)],cwd=ROOT,check=True)
print('Finished. Open dashboard/index.html. The report PDF and editable Overleaf project are linked in README.md.')

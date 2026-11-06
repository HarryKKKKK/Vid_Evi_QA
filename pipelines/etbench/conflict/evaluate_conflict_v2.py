#!/usr/bin/env python3
# vid-evi-compat-target: scripts/src/vid_evi_qa/evaluation/evaluate_conflict_v2.py
from pathlib import Path as _CompatPath
_compat_source = _CompatPath(__file__).resolve().parents[3] / 'scripts/src/vid_evi_qa/evaluation/evaluate_conflict_v2.py'
__file__ = str(_compat_source)
exec(compile(_compat_source.read_bytes(), __file__, 'exec'), globals(), globals())

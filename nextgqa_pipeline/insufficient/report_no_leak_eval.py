#!/usr/bin/env python3
"""Compatibility entry point. Edit pipelines/nextgqa/insufficient/report_no_leak_eval.py instead."""
# vid-evi-compat-target: pipelines/nextgqa/insufficient/report_no_leak_eval.py
from pathlib import Path as _CompatPath
_compat_source = _CompatPath(__file__).resolve().parents[2] / 'pipelines/nextgqa/insufficient/report_no_leak_eval.py'
if not _compat_source.is_file():
    raise RuntimeError("Missing canonical implementation: " + str(_compat_source))
__file__ = str(_compat_source)
exec(compile(_compat_source.read_bytes(), __file__, "exec"), globals(), globals())

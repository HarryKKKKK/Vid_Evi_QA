"""Copy implementation sources, not compatibility launchers, into frozen releases."""
from pathlib import Path
import re
import shutil

def resolve_script(source):
    source = Path(source)
    if source.suffix != '.py':
        return source
    with source.open('rb') as stream:
        header = stream.read(2048).decode('utf-8', errors='replace')
    match = re.search(r'^# vid-evi-compat-target: (.+)$', header, re.MULTILINE)
    if match is None:
        return source
    relative = Path(match.group(1).strip())
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid compatibility source: ' + str(relative))
    for parent in source.resolve().parents:
        target = parent / relative
        if target.is_file():
            return target
    raise FileNotFoundError('Cannot freeze compatibility entry without its implementation: ' + str(source))

def copy_script(source, destination, *, follow_symlinks=True):
    return shutil.copy2(resolve_script(source), destination, follow_symlinks=follow_symlinks)

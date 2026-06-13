import argparse
import multiprocessing as mp
import os
from functools import partial

import nncore


def _proc(blob, fps, size):
    nncore.mkdir(nncore.dir_name(blob[1]))
    cmd = f'ffmpeg -y -i {blob[0]} -filter:v "scale=\'if(gt(a,1),trunc(oh*a/2)*2,{size})\':\'if(gt(a,1),{size},trunc(ow*a/2)*2)\'" -map 0:v -r {fps} {blob[1]}'
    nncore.exec(cmd)
    assert nncore.is_file(blob[1]) and nncore.get_size(blob[1]) >= 1, blob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='videos')
    parser.add_argument('--tgt_dir', default='videos_compressed')
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.src_dir != args.tgt_dir

    blobs = []
    for pattern in ('*.mp4', '*.mkv', '*.webp'):
        for path in nncore.find(args.src_dir, pattern):
            tgt_path = nncore.join(args.tgt_dir, f'{nncore.split_ext(path[len(args.src_dir) + 1:])[0]}.mp4')
            blobs.append((path, tgt_path))

    if len(blobs) == 0:
        raise FileNotFoundError(f"No videos found in '{args.src_dir}'.")

    print(f'Total number of videos: {len(blobs)}')
    for b in blobs[:3]:
        print(b)

    print(f'Starting {os.cpu_count() if args.workers is None else args.workers} workers...')
    proc = partial(_proc, fps=args.fps, size=args.size)

    prog_bar = nncore.ProgressBar(num_tasks=len(blobs))
    with mp.Pool(args.workers) as pool:
        for _ in pool.imap_unordered(proc, blobs):
            prog_bar.update()

#!/usr/bin/env python3
"""Strip optimizer state from a `flow_step*.pt` checkpoint to produce an
`inference_step*.pt` suitable for deploy.

The `model` field in `flow_step*.pt` is already `_trainable_state_dict`
(frozen encoders like CLIP / DINOv2 are not saved), so this is purely a
drop of the `optimizer` entry plus a filename swap.

Usage:
    python convert_to_inference_ckpt.py logs/<run>/checkpoints/flow_step030000.pt
    python convert_to_inference_ckpt.py logs/<run>/checkpoints/flow_step*.pt
    python convert_to_inference_ckpt.py path/to/flow_step030000.pt -o custom_name.pt
"""

import argparse
import os
import re

import torch


def convert(src: str, dst: str | None = None) -> str:
    if dst is None:
        base = os.path.basename(src)
        new_base = re.sub(r'^flow_step', 'inference_step', base)
        if new_base == base:
            raise ValueError(
                f"Source filename '{base}' does not start with 'flow_step'; "
                f"pass an explicit --output path."
            )
        dst = os.path.join(os.path.dirname(src), new_base)

    ckpt = torch.load(src, map_location='cpu')
    out = {
        'step': ckpt.get('step'),
        'loss': ckpt.get('loss'),
        'model': ckpt['model'],
        'model_config': ckpt.get('model_config'),
    }
    torch.save(out, dst)

    src_mb = os.path.getsize(src) / 1024 / 1024
    dst_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"  {src}  ({src_mb:.1f} MB)")
    print(f"  -> {dst}  ({dst_mb:.1f} MB)")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='one or more flow_step*.pt files')
    parser.add_argument('-o', '--output', default=None,
                        help='output path (only valid with a single input)')
    args = parser.parse_args()

    if args.output is not None and len(args.inputs) != 1:
        parser.error('--output can only be used with a single input file')

    for src in args.inputs:
        convert(src, args.output)


if __name__ == '__main__':
    main()

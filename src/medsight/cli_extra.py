Extended medsight CLI commands.
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from medsight import datasets, preprocess

def _cmd_list_series(args):
    from medsight.datasets import dicom
    try:
        data, meta = dicom.load_dicom(args.path)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    info = {"shape": list(data.shape), "dtype": str(data.dtype), "modality": meta.get("Modality", "?"), "meta_keys": list(meta.keys())[:10]}
    print(json.dumps(info, indent=2))
    return 0

def build_extra_parser(sub):
    p = sub.add_parser("dicom-info", help="Read metadata from a DICOM file")
    p.add_argument("path", type=Path); p.set_defaults(func=_cmd_list_series)
    return p

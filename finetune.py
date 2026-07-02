"""
Fine-tune the detector on a project-specific (stray-dog) dataset.

Do NOT fine-tune on COCO — the stock YOLO26/YOLO11 weights were already
trained on it, so you would spend days of GPU time reproducing the weights
you started with. Fine-tuning only pays off on a *custom* dataset: your own
CCTV/ESP32 footage, or a stray-dog dataset exported in YOLO format (see the
"Fine-tuning" section in README.md for sources).

Expected dataset layout (standard Roboflow "YOLO" export):

    my_dataset/
    ├── data.yaml          <- paths + class names
    ├── train/images  + train/labels
    ├── valid/images  + valid/labels
    └── test/images   + test/labels   (optional)

Usage:

    # quick end-to-end sanity check (~7 MB download, minutes, proves training works)
    python finetune.py --data coco128.yaml --model yolo26n.pt --epochs 5

    # real fine-tune on your stray-dog dataset
    python finetune.py --data path/to/my_dataset/data.yaml --model yolo26n.pt --epochs 60

Defaults are sized for a laptop GPU (~4 GB VRAM, e.g. RTX 3050): batch 8,
imgsz 640. If you hit a CUDA out-of-memory error, drop --batch to 4, or use
--batch -1 to let Ultralytics auto-pick the largest batch that fits.
"""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO26/YOLO11 on a custom stray-dog dataset")
    p.add_argument("--data", required=True,
                   help="dataset data.yaml (or a built-in like coco128.yaml)")
    p.add_argument("--model", default="yolo26n.pt",
                   help="starting weights (yolo26n/s/m or yolo11n/m; "
                        "n/s recommended on 4 GB GPUs)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8,
                   help="8 fits ~4 GB VRAM; -1 = auto-pick by VRAM")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None,
                   help="0 for first GPU, 'cpu' to force CPU (default: auto)")
    p.add_argument("--name", default="straydog_finetune",
                   help="run name under the project dir")
    p.add_argument("--project", default=str(Path(__file__).parent / "runs" / "detect"),
                   help="where results are saved (default: THIS repo's runs/detect/, "
                        "overriding any global Ultralytics runs_dir from other projects)")
    p.add_argument("--patience", type=int, default=20,
                   help="early-stop after N epochs without improvement")
    p.add_argument("--workers", type=int, default=2 if os.name == "nt" else 8,
                   help="dataloader worker processes. Kept low on Windows by "
                        "default: each worker loads the CUDA DLLs and commits "
                        "GBs of virtual memory, so 8 workers triggers "
                        "'WinError 1455: paging file too small'")
    args = p.parse_args()

    stem = Path(args.model).stem.lower()
    if stem.endswith(("l", "x")) and stem.startswith(("yolo26", "yolo11")):
        print(f"[warn] {args.model} is a large model — fine-tuning it needs "
              f"~3x the memory of inference and OOMs on 8 GB laptop GPUs. "
              f"Prefer the s/m size (yolo26s.pt / yolo26m.pt); accuracy after "
              f"fine-tuning is usually close.")

    data = args.data
    # Built-in datasets like coco128.yaml have no path separator — Ultralytics
    # auto-downloads those. Anything that looks like a path must really exist.
    if ("/" in data or "\\" in data) and not Path(data).exists():
        raise SystemExit(
            f"\n--data '{data}' does not exist on this computer.\n\n"
            f"This must be the data.yaml INSIDE a dataset you have already\n"
            f"downloaded and extracted (see 'Fine-tuning' in README.md):\n"
            f"  1. universe.roboflow.com -> search 'stray dog' -> pick a dataset\n"
            f"  2. Download Dataset -> format YOLOv11 -> download the zip\n"
            f"  3. Extract the zip and note where data.yaml actually is\n"
            f"  4. Re-run with that real path, e.g.:\n"
            r"     python finetune.py --data C:\Users\HP\Downloads\<extracted-folder>\data.yaml"
            f" --model yolo26s.pt --epochs 60 --batch 16\n\n"
            f"Tip: Shift + right-click data.yaml in Explorer -> 'Copy as path'.\n")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Fine-tuning complete")
    print("=" * 60)
    print(f"  Best weights : {best}")
    print(f"  Results dir  : {results.save_dir}")
    print()
    print("  Use it in the apps:")
    print(f"    - Web app  : paste the path into 'Custom weights' in the sidebar")
    print(f"    - Desktop  : paste the path into 'Custom weights' in section 4")
    print(f"    - Config   : set detector.model: {best}")
    print("=" * 60)


if __name__ == "__main__":
    main()

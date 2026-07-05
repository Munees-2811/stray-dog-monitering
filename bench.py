"""
Speed benchmark — measures REAL inference FPS with no GUI.

This isolates the model from the app. If bench.py is fast but the app feels
slow, the problem is display/playback (fixed in app3). If bench.py itself is
slow, the model/size/pose or the torch install is the bottleneck.

Run against the SAME video and settings you use in the app:

    python bench.py --source "C:\\Users\\HP\\Videos\\clip.mp4" --model yolo26x.pt --imgsz 640 --pose
    python bench.py --source "C:\\Users\\HP\\Videos\\clip.mp4" --model yolo26n.pt --imgsz 320

It prints the compute device and the true frames-per-second after warmup.
"""

import argparse
import time

import cv2


def main():
    p = argparse.ArgumentParser(description="Measure real inference FPS")
    p.add_argument("--source", required=True, help="video path")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--pose", action="store_true", help="also run the pose model")
    p.add_argument("--frames", type=int, default=120, help="frames to time")
    args = p.parse_args()

    from src.pipeline import StrayDogMonitor

    pose = "yolo11n-pose.pt" if args.pose else None
    print(f"Loading {args.model} (imgsz={args.imgsz}, pose={'on' if pose else 'off'}) …")
    mon = StrayDogMonitor(detector_path=args.model, pose_model=pose,
                          imgsz=args.imgsz)
    print(f"Compute: {mon.compute}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")

    print("Warming up …")
    mon.detector.detect(frame)   # pay CUDA kernel-compile cost off the clock

    print(f"Timing {args.frames} frames …")
    t0 = time.time()
    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        mon.process_frame(frame)
        n += 1
    dt = time.time() - t0
    cap.release()

    fps = n / dt if dt > 0 else 0.0
    print("\n" + "=" * 54)
    print(f"  {n} frames in {dt:.2f}s  ->  {fps:.1f} FPS")
    print(f"  model={args.model}  imgsz={args.imgsz}  pose={'on' if pose else 'off'}")
    print("=" * 54)
    if fps < 5:
        print("  Very slow. If Compute says CPU, install CUDA torch (README).")
        print("  If it says GPU, use a smaller model/imgsz: try")
        print("    python bench.py --source ... --model yolo26n.pt --imgsz 320")
    elif fps < 20:
        print("  Usable with real-time playback. For higher FPS: yolo26n/s or imgsz 512.")
    else:
        print("  Fast. If the app still feels slow, pull the latest app3.py.")


if __name__ == "__main__":
    main()

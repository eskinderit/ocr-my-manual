#!/usr/bin/env python3
"""
detect_layout.py

Run a DocLayout-YOLO model (or any YOLO-compatible model via `ultralytics`) on images
in the `imgs/` directory, draw visible bounding boxes, and save annotated images to
`outputs/layouts/`. Also saves a JSON file with detections.

Usage:
    python -m vespa_retrieval.detect_layout --model PATH_OR_MODEL_NAME

Dependencies:
  pip install ultralytics pillow numpy

If you don't have a DocLayout-YOLO model file, you can pass a pretrained yolov8 model
name like `yolov8n.pt` if available, or provide a path to the model file.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

try:
    from doclayout_yolo import YOLOv10
except Exception:
    YOLOv10 = None

try:
    from ultralytics import YOLO as UltraYOLO
except Exception:
    UltraYOLO = None

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def find_images(img_dir: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def draw_boxes(image: Image.Image, boxes: np.ndarray, classes: List[str], scores: List[float], class_ids: List[int]):
    # Robust class name access
    def _class_name(cid):
        try:
            if isinstance(classes, dict):
                return classes.get(cid, str(cid))
            else:
                return classes[cid] if 0 <= cid < len(classes) else str(cid)
        except Exception:
            return str(cid)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # draw on an RGBA overlay for consistent visibility, then composite onto the image
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    for (x1, y1, x2, y2), cid, sc in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        outline = (255, 0, 0, 255)
        fill = (255, 0, 0, 48)  # translucent fill
        # thicker outline to ensure visibility
        odraw.rectangle([x1, y1, x2, y2], outline=outline, fill=fill, width=6)
        label_text = f"{_class_name(cid)} {sc:.2f}"
        # compute text size robustly for different PIL versions
        try:
            if font and hasattr(font, 'getsize'):
                tw, th = font.getsize(label_text)
            else:
                tb = odraw.textbbox((0,0), label_text, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
        except Exception:
            tw, th = (len(label_text) * 6, 11)
        text_size = (tw, th)
        text_bg = [x1, max(0, y1 - text_size[1] - 6), x1 + text_size[0] + 6, y1]
        odraw.rectangle(text_bg, fill=outline)
        odraw.text((x1 + 3, max(0, y1 - text_size[1] - 4)), label_text, fill=(255, 255, 255, 255), font=font)

    composed = Image.alpha_composite(base, overlay)
    # return the annotated RGB image
    return composed.convert('RGB')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='doclayout-yolo.pt', help='Path or name of YOLO model')
    p.add_argument('--imgs', type=str, default='imgs', help='Directory with input images')
    p.add_argument('--out', type=str, default='outputs/layouts', help='Directory to save annotated outputs')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.20, help='IoU threshold for deduplication (keep higher-confidence box)')
    p.add_argument('--imgsz', type=int, default=1024, help='Inference image size')
    args = p.parse_args(argv)

    img_dir = Path(args.imgs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    # Prefer doclayout_yolo's YOLOv10 if installed
    if YOLOv10 is not None:
        print('Using doclayout_yolo.YOLOv10 for inference')
        use_doclayout = True
        model = None
        try:
            # Prefer from_pretrained for HF repo ids
            if model_path and Path(model_path).is_file():
                model = YOLOv10(model_path)
            elif model_path and (model_path.startswith('juliozhao') or model_path.startswith('hf:') or '/' in model_path):
                try:
                    model = YOLOv10.from_pretrained(model_path)
                except Exception:
                    # fallback: try downloading a known checkpoint file from the repo
                    from huggingface_hub import hf_hub_download
                    possible_files = [
                        'doclayout_yolo_docstructbench_imgsz1024.pt',
                        'doclayout_yolo_docstructbench.pt',
                        'doclayout_yolo_doclaynet_imgsz1120.pt',
                    ]
                    for fname in possible_files:
                        try:
                            print(f'Trying to download {fname} from {model_path}...')
                            filepath = hf_hub_download(repo_id=model_path, filename=fname)
                            model = YOLOv10(filepath)
                            break
                        except Exception:
                            model = None
                    if model is None:
                        raise
            elif model_path and Path(model_path).exists():
                model = YOLOv10(model_path)
            else:
                try:
                    model = YOLOv10.from_pretrained('juliozhao/DocLayout-YOLO-DocStructBench')
                except Exception:
                    from huggingface_hub import hf_hub_download
                    filepath = hf_hub_download(repo_id='juliozhao/DocLayout-YOLO-DocStructBench', filename='doclayout_yolo_docstructbench_imgsz1024.pt')
                    model = YOLOv10(filepath)
        except FileNotFoundError as e:
            print('Model file missing:', e)
            return
        except Exception as e:
            print('Failed to load doclayout_yolo model:', e)
            return

        names = getattr(model, 'class_names', None) or getattr(model, 'names', None) or {i: str(i) for i in range(100)}
    elif UltraYOLO is not None:
        print('Using ultralytics.YOLO for inference')
        if not Path(model_path).exists():
            print(f"Model file {model_path} not found locally. Attempting to load by name if available.")
        model = UltraYOLO(model_path)
        names = None
        if hasattr(model, 'names') and model.names:
            names = model.names
        else:
            try:
                names = model.model.names
            except Exception:
                names = {i: str(i) for i in range(100)}
        use_doclayout = False
    else:
        print('Error: neither `doclayout_yolo` nor `ultralytics` is installed. Install one of them in your environment.')
        return

    images = find_images(img_dir)
    if not images:
        print(f'No images found in {img_dir}.')
        return

    all_detections = {}

    images = sorted(images, key=lambda p: p.name)

    for img_path in images:
        print(f'Running inference on {img_path.name}...')
        try:
            if use_doclayout:
                det_res = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, device='cpu')
                if not det_res:
                    print('No result for', img_path)
                    continue
                # we'll draw and save an annotated image after deduplication

                # extract boxes
                boxes = []
                scores = []
                class_ids = []
                dets = []
                for item in det_res[0].boxes:
                    # item.xyxy may be a tensor, numpy array, or nested array. Coerce to 1D numpy.
                    xy = getattr(item, 'xyxy', None)
                    if xy is None:
                        try:
                            xy = item.xyxy.cpu().numpy()
                        except Exception:
                            continue
                    xy = np.array(xy)
                    if xy.size == 0:
                        continue
                    xy = xy.ravel()
                    if xy.size < 4:
                        continue
                    x1, y1, x2, y2 = map(float, xy[:4])
                    # conf and cls may also be tensors/arrays
                    conf = getattr(item, 'conf', None)
                    cls = getattr(item, 'cls', None)
                    try:
                        sc = float(conf)
                    except Exception:
                        try:
                            sc = float(np.array(conf).ravel()[0])
                        except Exception:
                            sc = 0.0
                    try:
                        cid = int(cls)
                    except Exception:
                        try:
                            cid = int(np.array(cls).ravel()[0])
                        except Exception:
                            cid = -1

                    label = names.get(cid, str(cid)) if isinstance(names, dict) else (names[cid] if (isinstance(names, (list, tuple)) and cid < len(names) and cid >= 0) else str(cid))
                    boxes.append([x1, y1, x2, y2])
                    scores.append(sc)
                    class_ids.append(cid)
                    dets.append({'label': label, 'class_id': cid, 'score': sc, 'bbox': [x1, y1, x2, y2]})
                # helper: IoU and dedup functions
                def _iou(bb1, bb2):
                    x1, y1, x2, y2 = bb1
                    x3, y3, x4, y4 = bb2
                    xi1 = max(x1, x3)
                    yi1 = max(y1, y3)
                    xi2 = min(x2, x4)
                    yi2 = min(y2, y4)
                    iw = max(0, xi2 - xi1)
                    ih = max(0, yi2 - yi1)
                    inter = iw * ih
                    area1 = max(0, x2 - x1) * max(0, y2 - y1)
                    area2 = max(0, x4 - x3) * max(0, y4 - y3)
                    union = area1 + area2 - inter
                    return (inter / union) if union > 0 else 0.0

                def _dedup(dets_list, iou_thresh):
                    # keep boxes with highest score, skip boxes that overlap kept boxes above threshold
                    sorted_dets = sorted(dets_list, key=lambda d: float(d.get('score', 0.0)), reverse=True)
                    kept = []
                    for d in sorted_dets:
                        bb = tuple(map(float, d.get('bbox', [0,0,0,0])))
                        if any(_iou(bb, tuple(map(float,k.get('bbox', [0,0,0,0])))) > iou_thresh for k in kept):
                            continue
                        kept.append(d)
                    return kept

                # compute pairwise IoU among detections and print it
                if len(dets) > 1:
                    print(f'Pairwise IoU for {img_path.name}:')
                    for i in range(len(dets)):
                        for j in range(i+1, len(dets)):
                            bb1 = tuple(map(float, dets[i].get('bbox', [0,0,0,0])))
                            bb2 = tuple(map(float, dets[j].get('bbox', [0,0,0,0])))
                            iou_val = _iou(bb1, bb2)
                            print(f"  [{i}] {dets[i].get('label','')} vs [{j}] {dets[j].get('label','')}: {iou_val:.4f}")

                dets = _dedup(dets, args.iou)
                # draw annotated image with kept boxes
                try:
                    img_annot = Image.open(img_path).convert('RGB')
                    kept_boxes = [tuple(map(float, d.get('bbox', [0,0,0,0]))) for d in dets]
                    kept_scores = [float(d.get('score', 0.0)) for d in dets]
                    kept_cids = [int(d.get('class_id', -1)) for d in dets]
                    img_annot = draw_boxes(img_annot, kept_boxes, names if isinstance(names, dict) else list(names), kept_scores, kept_cids)
                    out_file = out_dir / img_path.name
                    img_annot.save(out_file)
                except Exception as e:
                    print(f"Failed to write annotated image for {img_path.name}:", e)
                    try:
                        # fallback: save original image so user has a file to inspect
                        out_file = out_dir / img_path.name
                        Image.open(img_path).convert('RGB').save(out_file)
                        print(f"Saved original image as fallback: {out_file}")
                    except Exception as e2:
                        print(f"Fallback save also failed for {img_path.name}:", e2)
                all_detections[img_path.name] = dets
                # write per-image detections JSON next to annotated image
                try:
                    per_path = out_dir / (img_path.name.replace('.png', '').replace('.jpg', '') + '.json')
                    with open(per_path, 'w', encoding='utf-8') as fh:
                        json.dump(dets, fh, indent=2)
                except Exception as e:
                    print(f'Failed to write per-image JSON for {img_path.name}:', e)
            else:
                results = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
                if len(results) == 0:
                    print('No result for', img_path)
                    continue
                res = results[0]
                boxes = []
                scores = []
                class_ids = []
                if hasattr(res, 'boxes') and res.boxes is not None:
                    try:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        try:
                            xyxy = np.array(res.boxes.xyxy)
                            confs = np.array(res.boxes.conf)
                            cls_ids = np.array(res.boxes.cls).astype(int)
                        except Exception:
                            xyxy = np.zeros((0, 4))
                            confs = np.zeros((0,))
                            cls_ids = np.zeros((0,), dtype=int)
                    boxes = xyxy.tolist()
                    scores = confs.tolist()
                    class_ids = cls_ids.tolist()

                # we'll draw and save annotated image after deduplication

                dets = []
                for (x1, y1, x2, y2), cid, sc in zip(boxes, class_ids, scores):
                    label = names.get(cid, str(cid)) if isinstance(names, dict) else (names[cid] if cid < len(names) else str(cid))
                    dets.append({'label': label, 'class_id': int(cid), 'score': float(sc), 'bbox': [float(x1), float(y1), float(x2), float(y2)]})

                # helper: IoU and dedup functions
                def _iou(bb1, bb2):
                    x1, y1, x2, y2 = bb1
                    x3, y3, x4, y4 = bb2
                    xi1 = max(x1, x3)
                    yi1 = max(y1, y3)
                    xi2 = min(x2, x4)
                    yi2 = min(y2, y4)
                    iw = max(0, xi2 - xi1)
                    ih = max(0, yi2 - yi1)
                    inter = iw * ih
                    area1 = max(0, x2 - x1) * max(0, y2 - y1)
                    area2 = max(0, x4 - x3) * max(0, y4 - y3)
                    union = area1 + area2 - inter
                    return (inter / union) if union > 0 else 0.0

                def _dedup(dets_list, iou_thresh):
                    sorted_dets = sorted(dets_list, key=lambda d: float(d.get('score', 0.0)), reverse=True)
                    kept = []
                    for d in sorted_dets:
                        bb = tuple(map(float, d.get('bbox', [0,0,0,0])))
                        if any(_iou(bb, tuple(map(float,k.get('bbox', [0,0,0,0])))) > iou_thresh for k in kept):
                            continue
                        kept.append(d)
                    return kept

                # compute pairwise IoU among detections and print it
                if len(dets) > 1:
                    print(f'Pairwise IoU for {img_path.name}:')
                    for i in range(len(dets)):
                        for j in range(i+1, len(dets)):
                            bb1 = tuple(map(float, dets[i].get('bbox', [0,0,0,0])))
                            bb2 = tuple(map(float, dets[j].get('bbox', [0,0,0,0])))
                            iou_val = _iou(bb1, bb2)
                            print(f"  [{i}] {dets[i].get('label','')} vs [{j}] {dets[j].get('label','')}: {iou_val:.4f}")

                dets = _dedup(dets, args.iou)
                # draw annotated image with kept boxes
                try:
                    img_annot = Image.open(img_path).convert('RGB')
                    kept_boxes = [tuple(map(float, d.get('bbox', [0,0,0,0]))) for d in dets]
                    kept_scores = [float(d.get('score', 0.0)) for d in dets]
                    kept_cids = [int(d.get('class_id', -1)) for d in dets]
                    img_annot = draw_boxes(img_annot, kept_boxes, names if isinstance(names, dict) else list(names), kept_scores, kept_cids)
                    out_file = out_dir / img_path.name
                    img_annot.save(out_file)
                except Exception as e:
                    print(f"Failed to write annotated image for {img_path.name}:", e)
                    try:
                        out_file = out_dir / img_path.name
                        Image.open(img_path).convert('RGB').save(out_file)
                        print(f"Saved original image as fallback: {out_file}")
                    except Exception as e2:
                        print(f"Fallback save also failed for {img_path.name}:", e2)
                all_detections[img_path.name] = dets
                try:
                    per_path = out_dir / (img_path.name + '.json')
                    with open(per_path, 'w', encoding='utf-8') as fh:
                        json.dump(dets, fh, indent=2)
                except Exception as e:
                    print(f'Failed to write per-image JSON for {img_path.name}:', e)
        except Exception as e:
            print(f'Error processing {img_path.name}:', e)
            all_detections[img_path.name] = {'error': str(e)}
            continue

    # Also write combined metadata for compatibility
    try:
        json_path = out_dir.parent / 'layouts.json'
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(all_detections, fh, indent=2)
        print(f'Finished. Annotated images saved to {out_dir}, metadata saved to {json_path} and per-image JSONs in {out_dir}')
    except Exception as e:
        print('Finished. Annotated images saved to', out_dir, 'but failed writing combined JSON:', e)


if __name__ == '__main__':
    main()

"""P2 — Détection (Q4-Q6).

Sanity YOLOv8n out-of-the-box + skeleton de fine-tuning sur dataset substitué.

Q4 fine-tune : voir `train_yolo_on_substituted(...)` (à exécuter sur Colab T4
si pas de MPS local utilisable). En local Apple Silicon : MPS disponible
(Q4 entraînement léger possible : YOLOv8n / epochs=20 / imgsz=640).

Q5 inférence + precision/recall sur 5 images du test split.
Q6 optimisation : albumentations + grid lr/batch/imgsz.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from . import config as C


def _device() -> str:
    """Détecte le device : CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------
# Sanity check — preuve de vie YOLOv8n out-of-the-box

def sanity_yolov8n(image_path: str | Path, model_name: str = "yolov8n.pt") -> dict:
    """Inférence YOLOv8n pré-entraîné COCO sur une image quelconque.

    But : valider la chaîne (modèle + libs) AVANT d'attaquer le fine-tune.
    YOLOv8 COCO ne connaît PAS « frégate »/« destroyer » — il connaît seulement
    `boat` (classe 8). On vérifie juste que le modèle tourne et trouve des objets.
    """
    image_path = Path(image_path)
    device = _device()
    model = YOLO(model_name)
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=640,
        conf=0.25,
        verbose=False,
        save=False,
    )
    r = results[0]
    boxes = r.boxes
    names = model.names

    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": round(conf, 3),
                "bbox_px": [round(v, 1) for v in (x1, y1, x2, y2)],
            }
        )

    # Annotated image
    out_img = C.OUTPUTS / f"sanity_yolo_{image_path.stem}.jpg"
    annotated = r.plot()
    try:
        from PIL import Image

        Image.fromarray(annotated[..., ::-1]).save(out_img)  # BGR -> RGB
    except ImportError:
        out_img = None

    return {
        "device": device,
        "model": model_name,
        "n_detections": len(detections),
        "detections": detections,
        "n_boats": sum(1 for d in detections if d["class_name"] in {"boat", "ship"}),
        "annotated_image": str(out_img.relative_to(C.ROOT)) if out_img else None,
        "_note": (
            "YOLOv8n entraîné COCO ne connaît que la classe `boat` (8) pour les navires. "
            "Pour distinguer frégate/destroyer/porte-avions, il faut un fine-tune sur "
            "xView3-SAR ou HRSC2016 (voir `train_yolo_on_substituted`)."
        ),
    }


# ----------------------------------------------------------------------
# Q4 — Fine-tune sur dataset substitué (squelette)

def train_yolo_on_substituted(
    data_yaml: str | Path,
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    model_name: str = "yolov8n.pt",
) -> dict:
    """Squelette de fine-tune YOLOv8 sur dataset substitué (xView3-SAR ou Airbus Ship).

    Args:
        data_yaml: chemin vers le fichier YAML Ultralytics (train/val/test + classes).
        epochs: nombre d'époques (20-50 selon dispo GPU).
        imgsz: taille d'image (640 par défaut, 1024 pour SAR plus précis).
        batch: batch size (ajuster selon RAM/VRAM).
        model_name: poids initiaux YOLOv8.

    Hypothèses (cf. hypotheses.md §B Q4) :
        - split 70/15/15, seed=42
        - optimiseur AdamW, lr0=1e-3
        - métrique principale : mAP@0.5 et mAP@[.5:.95]

    Returns:
        Dict avec chemins du modèle entraîné et métriques de validation.
    """
    device = _device()
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=C.SPLIT_SEED,
        optimizer="AdamW",
        lr0=1e-3,
        project=str(C.OUTPUTS / "yolo_runs"),
        name="finetune_substituted",
        exist_ok=True,
        verbose=True,
    )
    metrics = model.val(device=device, verbose=False)
    return {
        "best_weights": str(results.save_dir) + "/weights/best.pt",
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "device": device,
    }

# Q18 — Synthèse benchmark détecteurs & classifieurs

## Q16 — Détecteurs (inférence CPU, image sanity)

| model                                    |   n_iter |   median_ms |   p10_ms |   p90_ms |   n_detections | error                    |
|:-----------------------------------------|---------:|------------:|---------:|---------:|---------------:|:-------------------------|
| YOLOv8n (ultralytics, COCO)              |        3 |       26.05 |    24.99 |    27.51 |              6 | nan                      |
| RT-DETR-l (ultralytics, COCO)            |        3 |      357.97 |   356.53 |   363.8  |              9 | nan                      |
| Faster R-CNN R50-FPN (torchvision, COCO) |        3 |      559.13 |   549.4  |   564.65 |              5 | nan                      |
| Claude Vision (Anthropic) - skipped      |      nan |      nan    |   nan    |   nan    |            nan | ANTHROPIC_API_KEY absent |

## Q17 — Classifieurs binaires (5-fold CV sur features Q7)

| classifier   |   accuracy |   f1_militaire |   time_5cv_s |   n_samples |   n_features |
|:-------------|-----------:|---------------:|-------------:|------------:|-------------:|
| MLP          |     0.5039 |         0.6116 |        0.039 |         256 |           13 |
| RandomForest |     0.5078 |         0.5435 |        0.205 |         256 |           13 |
| SVM_RBF      |     0.4766 |         0.5347 |        0.025 |         256 |           13 |

## Limites assumées

- Pas de fine-tune local (CPU only) → mAP/AP des détecteurs non rapportés ici. 
  Pour Q4-Q6, voir `p2_detection.train_yolo_on_substituted` (à exécuter sur Colab).
- Classifieurs binaires : AUC ≈ 0.5 sur CSV synthétique (cf. P3-Q7) → toute valeur 
  rapportée mesure la **vitesse**, pas une qualité prédictive transposable.
- Faster R-CNN sur CPU : ~10× plus lent que YOLOv8n. RT-DETR-l : intermédiaire. 
  YOLOv8n reste la cible déploiement-friendly pour un pipeline temps réel.

## Pistes

- Q4-Q6 sur Colab T4 : fine-tune YOLOv8 sur xView3-SAR (recommandé) ou Airbus Ship. 
- Q8/Q9 : embeddings ResNet50 sur HRSC2016 + LogReg/RF + t-SNE des erreurs.
- Quantisation INT8 (ONNX Runtime) : décrochée car certaines couches Ultralytics 
  posent souci en quantisation dynamique — pister via `onnxruntime.tools` plus tard.
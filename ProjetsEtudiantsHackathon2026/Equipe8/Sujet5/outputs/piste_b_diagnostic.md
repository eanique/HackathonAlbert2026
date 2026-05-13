# Piste B — Diagnostic & résultats (chasse aux navires, imagerie réelle)

> Mise à jour : ajout d'un **détecteur SAR classique (CFAR à moyenne de cellules)**
> dans `src/hunt.py::detect_ships_cfar()`. Ne dépend d'aucun modèle entraîné →
> marche immédiatement sur Sentinel-1 RTC (≠ YOLO COCO qui échoue, cf. §B).

---

## A. Résultat actuel — CFAR sur Sentinel-1 RTC, 5 bases navales

Source : **Microsoft Planetary Computer** (STAC public, sans inscription), collection
`sentinel-1-rtc` (Radiometrically Terrain Corrected, ~10 m/pixel, polarisation VV).
Une scène récente (≤ 30 j) par base, fenêtre découpée sur la bbox de la base.

| Base navale | Scène Sentinel-1 | Date (UTC) | Cibles SAR | dont **eau libre** | dont portuaire | SNR médian |
|---|---|---|---:|---:|---:|---:|
| Toulon (FR) | S1A_…20260510T173047 | 2026-05-10 17:31 | 45 | **30** | 15 | 4.3 dB |
| Brest (FR) | S1A_…20260511T181323 | 2026-05-11 18:14 | 98 | **84** | 14 | 4.6 dB |
| Norfolk (US) | S1A_…20260506T225808 | 2026-05-06 22:58 | 21 | **11** | 10 | 4.3 dB |
| Severomorsk (RU) | S1C_…20260511T043020 | 2026-05-11 04:30 | 7 | **5** | 2 | 4.4 dB |
| Sébastopol (RU/UA) | S1C_…20260511T154326 | 2026-05-11 15:44 | 16 | **12** | 4 | 4.3 dB |
| **Total** | 5 scènes | — | **187** | **142** | 45 | — |

→ **187 cibles SAR brillantes détectées sur 5 zones de bases navales réelles, dont 142
candidats navires en eau libre** (vs **0** avec YOLO COCO). Données : `data/processed/chasse_*_s1.csv`
+ agrégat `chasse_toutes_bases_s1.csv`. Coût : **0 €**, ~30 s de download/scène (lecture
streaming du COG, pas la scène entière). Visibles dans le dashboard (`make dashboard-data` puis rafraîchir).

### Méthode CFAR (Cell-Averaging) — `detect_ships_cfar()`
1. Backscatter linéaire → dB (robuste aux spéculaires côtiers).
2. Pour chaque pixel : fond local = moyenne/écart-type de l'**anneau** = (fenêtre de fond
   `bg_px=25`) − (zone de garde `guard_px=5`), calculé par filtres de boîte (O(N), pas O(N·k²)).
3. Détection si `pixel_dB > μ_fond + k·σ_fond` avec `k = 3.5`.
4. Composantes connexes → blobs ; filtres : aire ∈ [3, 1500] px (rejette bruit & terre),
   **compacité** ≥ 0.30 (rejette les lignes = digues/quais), **fond local < percentile 55**
   de la scène (le blob doit être entouré de mer sombre).
5. Classification `target_kind` : `vessel_open_water` si ≥ 65 % des pixels dans un rayon
   de 25 px (≈ 250 m) sont sous le seuil "mer", sinon `harbor_target`.
6. Reprojection pixel → CRS source → WGS84 ; NMS global (les tuiles se recouvrent).

### Honnêteté méthodologique (à dire au jury / dans le rapport)
- Une "cible SAR eau libre" dans la bbox d'une base navale est un **candidat navire militaire
  à forte probabilité** (c'est tout l'intérêt de cibler ces périmètres) — mais ce n'est PAS
  un navire militaire **confirmé** : il faut soit du sub-métrique (Maxar 0.5 m / IGN BD ORTHO
  20 cm) pour identifier le type, soit un YOLO fine-tuné xView3, soit le croisement AIS
  (un candidat sans MMSI = "navire sombre", cf. `src/ais_cross.py`).
- Les "cibles portuaires" (45) sont navires **ou** infrastructure (quais, grues, digues) —
  CFAR sans land-mask ne les distingue pas. À ne pas compter comme navires.
- Brest domine (98) parce que la bbox couvre la rade + l'estuaire de la Penfeld + une partie
  de la ville → plus de structures. Filtrer plus serré = moins de FP mais on rate des navires
  au mouillage proches de la côte. Compromis assumé.

---

## B. Pourquoi YOLO COCO ne suffit pas (diagnostic conservé)

**Scène** : S2A MSIL2A 20260505 R108 T31TGH (Toulon, 12.6×11.5 km, 10 m/pixel, Sentinel-2 optique).
**Modèle** : YOLOv8n COCO pré-entraîné — PAS de fine-tune SAR/aérien.

**Résultat brut** (conf=0.05, imgsz=1600, upscale ×3) : **15 détections**, réparties en
`{'bear': 6, 'traffic light': 2, 'person': 5, 'sheep': 2}` — **aucune classée `boat`/`ship`**.

**Interprétation** : YOLO COCO a été entraîné sur des photos terrestres (vue de côté), pas
sur de l'imagerie satellite zénithale. Face à des navires alignés au mouillage dans la rade
militaire de Toulon, il projette les classes qu'il connaît. Image annotée :
`outputs/piste_b_toulon_yolo_coco_diag.jpg`.

→ Deux remèdes (les deux désormais disponibles dans `hunt.py`) :
1. **CFAR SAR classique** (§A) — branché, marche tout de suite, méthode de référence pour SAR.
2. **Fine-tune YOLO** sur xView3-SAR / Airbus Ship — interface `src.p2_detection.train_yolo_on_substituted`,
   exécution Colab T4 (~1-2 h). C'est ce qui débloquera la **classification de type** (frégate/destroyer/…).

---

## C. Ce que la Piste B prouve

- **Chaîne complète OK** : query STAC → signe l'asset (Planetary Computer) → lecture streaming
  rasterio (Cloud-Optimized GeoTIFF, fenêtre seulement) → conversion dB/uint8 → CFAR (ou tuilage+YOLO)
  → NMS global → reprojection pixel→WGS84 → CSV → dashboard.
- **Imagerie réelle de 5 bases navales récupérée et traitée** : Sentinel-1 RTC ~10 m/pix.
- **142 candidats navires en eau libre** sur ces 5 bases (Toulon, Brest, Norfolk, Severomorsk, Sébastopol).
- **0 €** (Planetary Computer STAC public).

## D. Prochaines étapes (cf. `DATA_NEXT_STEPS.md`)
- `GFW_API_TOKEN` → croiser les candidats avec les gap events AIS → liste de "navires sombres" confirmés.
- Colab T4 → `train_yolo_on_substituted('xview3.yaml', epochs=30)` → vraies classes de navires.
- Plusieurs scènes par base (séries temporelles) → détecter les variations d'effectif.
- Enrichissement : Wikipedia (composition de flotte attendue par base) pour le cross-check du pitch.

# Équipe — BDD-MinArm Sujet 5

## Composition
- *(à compléter : Prénom NOM — Albert School / École 42)*
- *(à compléter)*
- *(à compléter)*
- *(à compléter)*

## Sujet
**Sujet 5 — Traque de navires militaires par imagerie satellite.**

## Articulation avec l'autre équipe BDD-MinArm
L'équipe BDD-MinArm est verrouillée sur **2 sujets parallèles** :
- **Sujet 3** — Identification navale par signature radio (dossier `BDD-MinArm-Sujet3/`).
- **Sujet 5** — Traque de navires militaires par imagerie satellite (ce dossier).

**Récit commun au jury** : « rendre visible un navire qui veut rester invisible — par ses émissions radio (S3) et depuis l'orbite (S5). »

**Pont narratif** : Levier L2 « navire sombre » = une détection satellite **classée militaire**, **sans AIS** au même `(timestamp, bbox)` (croisement Danish Maritime Authority gratuit). Slide commune insérée dans les deux pitchs.

## Lancement rapide
Voir `README.md` du livrable + `docs/cadrage.md` (à reprendre tel quel depuis `sujet5/docs/`).

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env  # remplir les clés
python reponses_generalisation_detection_navires.py
```

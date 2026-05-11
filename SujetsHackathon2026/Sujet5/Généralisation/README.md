## QUESTIONS

--- 
### 📌 QUESTIONS POUR LA GÉNÉRALISATION : DÉTECTION ET CLASSIFICATION AVANCÉE DE NAVIRES

#### **🔹 Objectif**
Développer un outil complet pour détecter et classer des navires sur images satellites.
Utiliser les fichiers :
- `images_metadata_large.csv` (100 images avec métadonnées)
- `annotations_large.json` (annotations COCO)
- `military_zones.csv` (20 zones militaires)
- `detection_results.csv` (résultats de détection)
- Images satellites (à télécharger depuis les sources citées)

--- 

#### **📊 Partie 1 : Prétraitement et Exploration des Données**
1. **Analyse des métadonnées**
   - Quels sont les **3 sources d'images** les plus fréquentes dans `images_metadata_large.csv` ?
   - Quelle est la **résolution moyenne** des images ?
   - Combien d'images ont une **couverture nuageuse > 30%** ?

2. **Analyse des annotations**
   - Combien de **navires militaires** sont annotés dans `annotations_large.json` ?
   - Quelle est la **répartition par type de navire** (en %) ?
   - Quelle est la **taille moyenne des bounding boxes** pour chaque type de navire ?

3. **Fusion des données**
   - Fusionnez `images_metadata_large.csv` et `detection_results.csv` pour obtenir une vue complète des détections.
   - Combien de **détections par image** en moyenne ?
   - Quelle image a le **plus grand nombre de détections** ?

--- 

#### **🔍 Partie 2 : Détection d'Objets avec YOLO ou Faster R-CNN**
4. **Entraînement d'un modèle de détection**
   - Utilisez `annotations_large.json` pour entraîner un modèle de détection d'objets (ex: YOLOv8, Faster R-CNN).
   - Divisez les données en **train/val/test** (70%/15%/15%).
   - Quelle est la **précision (mAP)** de votre modèle sur l'ensemble de test ?

5. **Détection sur de nouvelles images**
   - Appliquez votre modèle à 5 images de `images_metadata_large.csv` **non utilisées pour l'entraînement**.
   - Comparez les résultats avec les annotations manuelles (`annotations_large.json`).
   - Calculez la **précision** et le **rappel** pour ces images.

6. **Optimisation du modèle**
   - Essayez d'améliorer votre modèle en :
     - Augmentant les données (ex: rotations, retournements, changements de luminosité).
     - Ajustant les hyperparamètres (ex: learning rate, batch size).
   - Quelle amélioration de **mAP** obtenez-vous ?

--- 

#### **🎯 Partie 3 : Classification des Navires (Militaires vs Civils)**
7. **Entraînement d'un classifieur binaire**
   - Extrayez les **features** des navires détectés (ex: taille du bounding box, ratio largeur/hauteur, position dans l'image).
   - Entraînez un classifieur binaire (ex: Random Forest, SVM, ou un CNN) pour distinguer les **navires militaires** des **navires civils**.
   - Quelle est la **précision** et le **rappel** de votre classifieur ?

8. **Classification multi-classes**
   - Entraînez un classifieur **multi-classes** pour prédire le type exact de navire (ex: Frégate, Destroyer, Porte-avions).
   - Utilisez les **features** extraites + les **embeddings** d'un modèle de vision (ex: ResNet50).
   - Quelle est l'**accuracy** de votre classifieur sur l'ensemble de test ?

9. **Analyse des erreurs de classification**
   - Identifiez les **fausses classifications** les plus fréquentes (ex: Frégate → Cargo).
   - Quelles sont les **caractéristiques communes** aux navires mal classés ?
   - Proposez des solutions pour réduire ces erreurs.

--- 

#### **🌍 Partie 4 : Analyse Géospatiale et Temporelle**
10. **Détection de navires dans des zones militaires**
    - Pour chaque détection dans `detection_results.csv`, vérifiez si elle se trouve dans une **zone militaire** (`military_zones.csv`).
    - Combien de **navires militaires** ont été détectés dans des zones militaires ?
    - Générez une **carte interactive** (avec `folium`) montrant ces détections.

11. **Analyse temporelle**
    - Pour chaque zone militaire, tracez l'**évolution du nombre de détections** sur le temps.
    - Y a-t-il des **pics d'activité** à certaines périodes ?
    - Corrélez ces pics avec des **événements géopolitiques** (ex: exercices militaires).

12. **Détection d'anomalies géospatiales**
    - Identifiez les **navires civils** détectés dans des zones militaires.
    - Identifiez les **navires militaires** détectés en dehors de zones militaires.
    - Ces détections sont-elles **suspectes** ? Pourquoi ?

--- 

#### **🤖 Partie 5 : Automatisation et Pipeline Complet**
13. **Pipeline de détection et classification**
    - Concevez un **pipeline complet** qui :
      1. Charge une nouvelle image satellite.
      2. Applique le modèle de **détection d'objets** pour localiser les navires.
      3. Applique le modèle de **classification** pour identifier le type de navire.
      4. Génère un **rapport** avec les résultats (positions, types, confiance).
      5. Envoie une **alerte** si un navire militaire est détecté dans une zone sensible.
    - Testez votre pipeline sur 10 nouvelles images.

14. **Optimisation des performances**
    - Mesurez le **temps d'exécution** de votre pipeline pour une image.
    - Proposez des optimisations pour :
      - Réduire le temps de traitement (ex: utiliser ONNX pour accélérer l'inférence).
      - Réduire la consommation mémoire (ex: quantisation du modèle).
      - Améliorer la précision (ex: fusion de modèles).

15. **Intégration avec des API externes**
    - Utilisez une **API de géolocalisation** (ex: Google Maps, OpenStreetMap) pour obtenir des informations supplémentaires sur les zones détectées.
    - Intégrez une **API météo** pour vérifier si les détections sont affectées par des conditions météorologiques (ex: brouillard).
    - Comment ces données externes peuvent-elles **améliorer la détection** ?

--- 

#### **📊 Partie 6 : Évaluation et Benchmarking**
16. **Benchmarking des modèles**
    - Comparez les performances de **3 modèles de détection d'objets** (ex: YOLOv8, Faster R-CNN, EfficientDet) sur votre jeu de données.
    - Quel modèle offre le **meilleur compromis précision/vitesse** ?
    - Justifiez votre choix.

17. **Évaluation de la classification**
    - Comparez les performances de **3 classifieurs** (ex: Random Forest, SVM, ResNet50) pour la classification des navires.
    - Quel classifieur est le **plus précis** pour distinguer les navires militaires ?
    - Quel classifieur est le **plus rapide** ?

18. **Rapport de performance global**
    - Générez un **rapport complet** avec :
      - Les métriques de performance (mAP, précision, rappel, F1-score).
      - Les temps d'exécution pour chaque étape du pipeline.
      - Les **limites** de votre solution (ex: faux positifs, faux négatifs).
      - Des **pistes d'amélioration** (ex: plus de données, modèles plus performants).

--- 

#### **📌 Consignes pour les Réponses**
- **Format des réponses** : Utilisez des bibliothèques comme `ultralytics` (YOLOv8), `torchvision` (Faster R-CNN), `scikit-learn` (classifieurs), et `folium` (cartes).
- **Exemple de code pour la détection d'objets** :
  ```python
  from ultralytics import YOLO
  import cv2
  
  # Charger un modèle YOLOv8 pré-entraîné
  model = YOLO("yolov8n.pt")
  
  # Appliquer le modèle à une image
  results = model("satellite_000.jpg")
  
  # Afficher les résultats
  for result in results:
      boxes = result.boxes  # Bounding boxes
      for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0].tolist()
          conf = box.conf[0].item()
          cls = box.cls[0].item()
          print(f"Détection: classe={cls}, confiance={conf:.2f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
  ```
- **Exemple de code pour la classification** :
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  import numpy as np
  
  # Exemple : Extraire des features (taille, ratio, etc.)
  X = np.random.rand(100, 5)  # 100 échantillons, 5 features
  y = np.random.randint(0, 2, 100)  # 0: civil, 1: militaire
  
  # Diviser en train/test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  
  # Entraîner un classifieur
  clf = RandomForestClassifier()
  clf.fit(X_train, y_train)
  
  # Évaluer
  accuracy = clf.score(X_test, y_test)
  print(f"Accuracy: {accuracy:.2f}")
  ```
- **Livrables** :
  - Un fichier `reponses_generalisation_detection_navires.py` avec le code.
  - Un rapport `rapport_generalisation_detection_navires.md` avec les résultats et analyses.


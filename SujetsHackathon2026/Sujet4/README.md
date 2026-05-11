## PROBLÈME
La flotte fantôme utilise des techniques de spoofing AIS pour contourner les sanctions internationales. Ces navires :
- Désactivent leur transpondeur AIS pour disparaître des radars.
- Modifient leur MMSI ou leur nom pour éviter la détection.
- Transmettent de fausses positions pour masquer leurs mouvements.

Problème : comment identifier en avance ces navires suspects à partir de données OSINT (AISHub, Marine Traffic) et de techniques d’IA ? Identifier automatiquement des comportements maritimes inhabituels ou potentiellement suspects à partir de données AIS.

## SOLUTION 
Concevoir une solution autonome, à partir d’outils open et sources de données OSINT, un pipeline automatisé pour :
- Collecter des données AIS en temps réel/historique.
- Nettoyer et analyser les données pour détecter les comportements suspects.
- Modéliser les relations entre navires, zones, et événements (graphe de connaissances).
- Visualiser les résultats via un tableau de bord interactif.
- Automatiser la génération d’alertes (email, rapport PDF).

## SOURCES
https://www.marinetraffic.com/

https://www.aishub.net

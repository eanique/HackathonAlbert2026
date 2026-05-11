## PROBLÈME

L’objectif de ce hackathon est de concevoir un outil capable d’exploiter des images satellites afin de détecter et de classifier des navires présents à la surface de la mer ou dans les ports.

Concrètement, les équipes devront développer une solution capable d’analyser automatiquement une image satellite et d’identifier la présence de bateaux. Une fois détectés, ces objets devront être localisés sur l’image et, dans la mesure du possible, classifiés selon leur nature. L’enjeu est notamment de pouvoir distinguer les navires civils des navires militaires, et d’identifier certains types caractéristiques comme les frégates, les destroyers ou les porte-avions.

## SOLUTION

Pour rendre l’exercice concret et stimulant, la mission proposée pendant ce hackathon prendra la forme d’une véritable opération de recherche. Les participants devront utiliser leur outil pour explorer différentes zones du globe et identifier le plus grand nombre possible de navires militaires visibles sur des images satellites accessibles publiquement. Les ports militaires, les bases navales ou certaines zones maritimes stratégiques pourront constituer des terrains d’exploration particulièrement intéressants.

L’objectif final est de montrer comment un outil basé sur l’intelligence artificielle peut transformer des images satellites brutes en information exploitable. L’équipe qui remportera la mission sera celle qui parviendra à identifier le plus grand nombre de navires militaires tout en démontrant la pertinence et l’efficacité de sa méthode de détection.

## SOURCE
Ressources pour les Images Satellites
Pour obtenir des images satellites réelles avec des navires, utilisez ces sources :

Navires avec labels : 
+ https://huggingface.co/datasets/DefendIntelligence/vessel-detection-labeled-patches


xView Dataset : https://xviewdataset.org/

Contient 1 million d'objets annotés sur des images satellites (dont des navires).
Format COCO compatible.


Sentinel Hub : https://www.sentinel-hub.com/

Accès aux images Sentinel-2 (résolution 10m, gratuites).
API pour télécharger des images.


Maxar Open Data Program : https://www.maxar.com/open-data

Images haute résolution (jusqu'à 0.5m) pour des événements spécifiques.


Google Earth Engine : https://earthengine.google.com/

Accès à des pétabytes d'images satellites (Sentinel, Landsat, etc.).
API Python pour l'analyse.


Kaggle Datasets :

Ships in Satellite Imagery : https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery

Contient des images avec des navires annotés.



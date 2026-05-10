**Avec les fichiers fournis dans ce répertoire :** 

+ Extrait CVE,
+ Extrait MITRE ATT&CK,
+ Mapping entre TTP et CVE,
+ Exemple de log linux, réseau et Windows.

1. Lire les fichiers log avec Python, et normaliser les données 
(pandas, json, https://pypi.org/project/attackcti/, https://pypi.org/project/pyattck/,
https://github.com/oasis-open/cti-python-stix2 (format STIX2...)

2. Extraire les TTPs selon divers critères (ID, nom, tactique, procédures, groupes d’attaquants, campagnes d’attaques, exemples de détection) pour cibler les TTP pertinentes. Préparer la clustérisation (NLP) les TTP similaires. 

3. Extraire les CVE selon divers critères (ID, score CVSS, logiciels affectés, exploitations). Préprarer le scoring des CVE selon le contexte métier (CVSS).

4. Mapper les CVE avec TTPs (ex. : CVE-2021-44228 → T1190). Préparer l'automatisation des liens (NLP, graphe de connaissance, ...).

5. Générer un rapport avec les TTPs prioritaires (par exemple : CVSS > 8)

L'objectif avec les briques ci-dessus,est d'aller vers la construction d'un pipeline automatisé :

1. Parse et normalise des logs bruts (Sysmon, etc.).
2. Enrichit les logs avec des métadonnées (TTPs, CVE, criticité).
3. Détecte des anomalies (ML, etc.).
4. Mappe automatiquement les logs aux TTPs/CVE avec NLP et graphes.
5. Génère des rapports prioritaires (% de log valides, CVSS > 8, TTPs critiques, etc.).

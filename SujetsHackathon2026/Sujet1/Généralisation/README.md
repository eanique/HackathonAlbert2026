**Avec les fichiers :**

+ Extraction MITRE ATT&CK du 7 mai, domaine “Entreprise”
  + Fichier fournit
  + Source : https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json

+ CVE : à récupérer
  + Format : sur github, CVEPproject/cve-schema
  + Source : https://www.cve.org/Downloads, cvelistv5-main.zip


1. Généralisez la démarche avec une extraction automatique quotidienne du fichier (utilisez le cas échéant l’API TAXII de MITRE) et une association TTP/CVE
2. Testez la démarche avec une base de logs plus réalistes synthétiques ou anonymisés 
Comment construire les relations TTP – CVE  ?
3. Explorez l’analyse textuelle (NLP), l’utilisation de bases de données tierces,…
4. Donner un score à la relation TTP/CVE : score de confiance, score d’impact sur l’environnement SI d’après les logs; …


**Avec la base de données suivante:**

1️⃣  Fichier nodes_attacks_final.csv

Contenu : 100 attaques (TTPs et CVE) représentatives.
Structure :

id (ex: T1059, CVE-2021-44228).
name, type (TTP/CVE), tactics, description, severity.
cvss_score et cvss_vector pour les CVE.


2️⃣ Fichier rels_dependencies_final.csv

Contenu : ~2 000 dépendances entre actifs (1 à 4 dépendances par actif).
Structure :

source, target (IDs des actifs).
type (DEPENDS_ON).
weight (0.5, 0.8, ou 1.0 pour indiquer l’importance).
description (ex: "L'actif IT-SRV-001 dépend de IT-NET-001 pour communiquer.").


3️⃣ Fichier rels_targets_final.csv

Contenu : ~500 liens entre attaques et actifs ciblés.
Structure :

source (ID de l’attaque, ex: T1059).
target (ID de l’actif, ex: IT-SRV-001).
type (TARGETS).
impact_score (0.5 à 1.0).
mitigation (ex: "Isoler IT-SRV-001 et appliquer les correctifs pour T1059.").


4️⃣ Fichier rels_generated_final.csv

Contenu : ~300 liens entre attaques et événements générés.
Structure :

source (ID de l’attaque).
target (ID de l’événement, ex: event-0001).
type (GENERATES).
probability (0.5 à 1.0).


5️⃣ Fichier nodes_events_final.csv

Contenu : 200 événements (incidents, alertes, attaques détectées, pannes, maintenances).
Structure :

id, name, type (Incident/Alerte/Attaque Détectée/Panne/Maintenance).
timestamp (format ISO 8601).
severity (critical, high, medium, low).
description, source (sysmon/apache/suricata/firewall/manual).
related_assets (liste des IDs d’actifs liés).
related_attacks (liste des IDs d’attaques liées, si applicable).


6️⃣ Fichier processes_final.csv

Contenu : Liste unique et exhaustive de tous les processus métier référencés dans vos fichiers.
Structure :

id (ex: process-001).
name (ex: "Gestion des Réservations").
description (ex: "Processus métier : Gestion des réservations clients et des contrats.").



Simuler de manière automatisée l’impact en cascade d’une panne ou d’une cyberattaque : identifier les processus métier affectés

Utiliser un LLM pour enrichir les métadonnées (ex: décrire automatiquement les actifs),


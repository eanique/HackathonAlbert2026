# SUJET N°1 : LOG AS CODE, GENERATION AUTOMATIQUE DE POLITIQUES DE LOG DE SECURITE

## PROBLÈME

Les organisations accumulent des millions de logs (sécurité, réseau, applications), mais leur analyse manuelle est chronophage, trop lente, sujette à l’erreur, et souvent inefficace face aux attaques modernes. Les équipes de sécurité ne savent pas toujours quels logs collecter pour détecter une menace spécifique. Comment automatiser la génération de politiques de logs pour :

+ Détecter les anomalies (ex. : attaques, erreurs de configuration).
+ Prioriser les alertes en fonction du risque.
+ Générer des règles de collecte et de rétention adaptées aux menaces.
+ Simuler l’impact d’une cyberattaque sur les systèmes critiques.

## SOLUTION

Utiliser des outils open source et des données OSINT pour :

### Sous-projet pour des équipes 100% Albert School
+ Analyser les logs existants et extraire des patterns normaux/anormaux.
+ Modéliser les menaces à partir de données ouvertes (ex. : MITRE ATT&CK, CVE, NVD,…).

### Sous-projet pour des équipes Albert School + étudiants de l’Ecole 42
+ Générer des politiques de logs (ex. : quels logs collecter, où les stocker, combien de temps les conserver).
+ Automatiser la détection et la réponse aux incidents.

## Consignes
Commencez par la section "Mise en jambe" pou comprendre le problème, le contexte, les outils, les données. 
Continuer avec la section "Généralisation". 

# SOURCES ET LIENS
## CTI
La Cyber Threat Intelligence définit la recherche, l’analyse et la modélisation de la menace cyber. 
Elle permet de décrire une menace ou une attaque informatique au travers d’éléments contextualisés et/ou 
d’indicateurs compréhensibles par des hommes ou des machines.
Avec l’augmentation de cyber attaques toujours plus sophistiquées, il est devenu primordial 
d’acquérir et de maintenir une connaissance de la menace et de l’attaquant pour les entreprises et les 
institutions. 

L'organisation pou laquelle vous traitez le sujet évolue dans un environnement technologique riche, 
complexe et distribué. Cet environnement comporte des _systèmes informatiques d'entreprise_, IT, et 
des _systèmes industriels connectés_, OT (terminaux intelligents, objets connectés, systèmes industriels). La sécurité IT et OT protège deux domaines distincts que les attaquants exploitent désormais comme une seule surface d’attaque. Le présent challenge se concentre sur les systèmes informatiques 
d'entreprise (IT).

## MITRE ATT&CK® 
MITRE ATT&CK is a globally-accessible knowledge base of adversary tactics and techniques based on real-world observations. 
The ATT&CK knowledge base is used as a foundation for the development of specific threat models and 
methodologies.
The "https://github.com/mitre-attack/mitreattack-python/#mitreattack-python" repository contains a library of Python tools and utilities for working with ATT&CK data. 

## TTP
Les TTP (Tactics, Techniques, and Procedures) décrivent comment les attaquants opèrent pour atteindre 
leurs objectifs, en combinant des objectifs stratégiques (Tactiques), des méthodes techniques (Techniques), 
et des implémentations concrètes (Procédures). Elles sont au cœur du framework MITRE ATT&CK et permettent 
une détection proactive des cybermenaces.

+ Site MITRE ATT&CK® : https://attack.mitre.org/ 

## STIX(TM) Version 2.0
STIX(TM) Version 2.0 is a language for expressing cyber threat and observable information. 
The web page https://www.oasis-open.org/standard/stix2-0/ defines concepts that apply across all of 
STIX and defines the overall structure of the STIX language.

# SUJET N°2 : Cartographie des impacts métier d'une attaque cyber

## PROBLÈME

Aujourd’hui, les organisations manquent d’un outil open source pour :

+ Cartographier automatiquement les dépendances entre actifs IT/OT et leurs processus métier.
+ Simuler l’impact d’une cyberattaque sur ces dépendances.
+ Visualiser les résultats de manière claire et actionnable pour les décideurs.
+ Automatiser la génération de rapports pour accélérer la réponse aux incidents.


## SOLUTION

Potentiellement requiert un soutien des étudiants de l’école 42

Utiliser des outils open source et des données OSINT pour :

+ Extraire et structure les données (logs, vulnérabilités, schémas réseau).
+ Construire un graphe de connaissances des actifs, vulnérabilités, et dépendances.
+ Simuler l’impact d’une attaque sur les processus métier.
+ Génèrer des visualisations et rapports pour aider à la prise de décision.

## Consignes
Commencez par la section "Mise en jambe" pour comprendre le problème, le contexte, les outils, les données. 
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

## CVSS Common Vulnerability Scoring System (CVSS)
The Common Vulnerability Scoring System (CVSS) is a method used to supply a qualitative measure of severity.

https://nvd.nist.gov/vuln-metrics/cvss

https://nvd.nist.gov/vuln-metrics/cvss/v2-calculator





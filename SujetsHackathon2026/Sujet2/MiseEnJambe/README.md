**À partir « neo4j dataset » fournit et contenant :**
+ Nodes assets : les actifs du SI
+ Nodes attacks : un exemple d’attaques (TTP/CVE)
+ Nodes events : les événements systèmes détectés, potentiellement suite à des attaques cyber
+ Rels dependencies : les liens entre actifs
+ Rels generated : les liens attaques/événements
+ Rels includes : les liens entre attaques
+ Rels targets : les liens entre attaques et actifs

En utilisant Neo4j ou NetworkX ou un autre produit de votre choix : 

1. Extraire les données, les normaliser, décire les schémas dans un dictionnaire de données,
1. Charger les fichiers dans un graphe de connaissance cohérent. Expliquer vos choix de modélisation,
1. Simuler l’impact (propagation) d’une attaque sur le graphe, identifier les calculer les impacts selon la criticité des actifs et/ou processus, 
1. Générer automatiquement un rapport d’impact détaillé accompagné de visualisations graphiques interactives, expédier le rapport par email. 

Projet MSPR — Dashboard Data & Machine Learning
Objectif du projet

Ce projet MSPR a pour objectif de mettre en place une chaîne complète de traitement de données électorales et socio-économiques comprenant :
Collecte et nettoyage des données (ETL)
Stockage en base de données
Analyse exploratoire (EDA)
Visualisation interactive avec Streamlit
Modélisation Machine Learning
Restitution sous forme de dashboard

Le projet permet notamment d’analyser les liens entre :
taux de chômage
abstention et participation
résultats électoraux
familles politiques

Arborescence du projet
MSPR/
│
├── data_clean/ # Données nettoyées prêtes à l’analyse
│ ├── chomage_2022_T3.csv
│ ├── elections_t1.csv
│ ├── elections_t2.csv
│ ├── table_finale.csv
│ ├── table_finale_famille.csv
│ ├── table_finale_HDF.csv
│ └── table_finale_HDF_famille.csv
│
├── data_raw/ # Données brutes
│
├── images/ # Résultats graphiques
Machine Learning
│ ├── ml_accuracy_comparison.png
│ ├── ml_confusion_matrices.png
│ ├── ml_feature_importance.png
│ └── ml_roc_curves.png
│
├── script/ # Scripts Python
│ ├── etl_cleaning.py
│ ├── generation_dump.py
│ └── model_ml.py
│
├── visualisation/ # Application Streamlit
│ └── app.py
│
├── database.db # Base de données SQLite
├── dump.sql # Dump SQL
└── README.md

Technologies utilisées
Python 3.12
Pandas — manipulation de données
Matplotlib / Seaborn — visualisation
Scikit-learn — Machine Learning
SQLite — base de données
Streamlit — dashboard interactif

Pipeline de données :
ETL & nettoyage
Script : script/etl_cleaning.py
Nettoyage des données brutes
Harmonisation des formats
Fusion des jeux de données
Génération de la base
Script : script/generation_dump.py
Création de la base SQLite
Export SQL (dump.sql)
Machine Learning
Script : script/model_ml.py
Entraînement des modèles
Génération des graphiques de performance
Visualisation
Application Streamlit : visualisation/app.py

Lancer l’application :
Installer les dépendances
pip install streamlit pandas matplotlib seaborn scikit-learn

Lancer le dashboard Streamlit
Depuis la racine du projet :
cd visualisation
streamlit run app.py

Fonctionnalités du dashboard
Accueil
Présentation du projet et de ses objectifs.

    Visualisations
        Histogrammes (abstention, participation, chômage)
        Heatmap de corrélation
        Scatter plots
        Boxplots par famille politique

    Exploration
        Sélection par département
        Indicateurs clés
        Graphiques dynamiques
        Tableau interactif

    Machine Learning
        Comparaison des modèles
        Matrices de confusion
        Importance des variables
        Courbes ROC

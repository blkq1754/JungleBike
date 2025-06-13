# 🚴‍♂️ Classification de Catégories de Produits Vélo

## Introduction

Solution et  Test Technique – Data Scientist chez Jungle Bike  : classification de produits de vélo à partir de texte. 

* **Démo en Ligne :** [Lien vers l'application Gradio sur Hugging Face Spaces](https://huggingface.co/spaces/belkacemsaad/demo_JungleBike)
* **Auteur :** Saad Belkacem

---

## Tableau Comparatif des Performances 

| Métrique | Modèle Classique (TF-IDF + LogReg) | Modèle Transformer (CamemBERT) | 
| :--- | :--- | :--- | :--- |
| **Précision | 98.0% | **99.0%** | 
| **F1-Score  | 98.0% | **99.0%** | 
| **Temps d'Entraînement** | **~ 5 minute** | ~ 3 heures (sur GPU) |  
| **Vitesse d'Inférence (CPU)** | **~ 11 IPS** | Lente | 


---

##  Structure du Projet

Le projet est organisé de manière modulaire pour une meilleure clarté et maintenabilité.

```
.
├── classic_ml/               # Pipeline du modèle TF-IDF + LogReg
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── data/                       # (non versionnées par Git)
├── evaluation/                 # Evaluation
│   ├── classic_model_results/
│   ├── transformer_model_results/
│   ├── evaluate_classic.py
│   └── evaluate_transformer.py
├── models/                     # Modèles 
│   ├── classic_model.joblib
│   ├── product_classifier.onnx
│   └── dataset_target_map.json
├── scripts/                    # Scripts d'action pour le pipeline Transformer
│   ├── __init__.py
│   ├── export_and_benchmark.py
│   ├── preprocess.py
│   ├── predict.py
│   └── train.py
├── src/                        
│   ├── __init__.py
│   └── ...
├── tests/                      # Tests unitaires
│   ├── __init__.py
│   └── test_*.py
├── app.py                      # app Gradio
├── pyproject.toml              # Fichier de configuration Poetry
└── README.md                   # Ce document
```

---

## ⚙️ Guide d'Installation et d'Utilisation

Ce guide détaille comment installer l'environnement et exécuter chaque script du projet.

###  Installation

Le projet utilise **Poetry** pour la gestion de l'environnement et des dépendances.

```bash
# 1. Cloner le dépôt
git clone https://github.com/blkq1754/Jungle-Bike.git
cd Jungle-Bike

# 2. Installer les dépendances
poetry install
```

###  Workflow

#### Tester l'Environnement

Ce script valide que la préparation des données fonctionne comme prévu.

```bash
poetry run pytest
```

#### Étape B : Pré-traitement dataset pour le Modèle Transformer

Ce script prépare et tokenize les données, nécessaire avant d'entraîner le modèle Transformer.

```bash
poetry run python -m scripts.preprocess
```

####  Entraînement des Modèles (Optionnel)

Les modèles étant déjà fournis, cette étape est optionnelle.

* **Entraîner le Modèle Classique (TF-IDF + LogReg) :**
    ```bash
    poetry run python -m classic_ml.train
    ```
* **Entraîner le Modèle CamemBERT (fine tuning) :**
    ```bash
    poetry run python -m scripts.train
    ```

#### Évaluation

Ces scripts génèrent les rapports de performance et les matrices de confusion dans le dossier `evaluation/`.

* **Évaluer le Modèle Classique :**
    ```bash
    poetry run python -m evaluation.evaluate_classic
    ```
* **Évaluer le Modèle Transformer :**
    ```bash
    poetry run python -m evaluation.evaluate_transformer
    ```

#### Lancer l'app gradio

```bash
poetry run python app.py
```

---

##  Démarche Technique 

### 1. Préparation du Dataset

* **Problème :** Incohérence des labels et distribution très déséquilibrée des catégories => Utiliser une source de vérité unique (le référentiel de catégories) pour garantir la fiabilité des labels.
*  Filtrer les catégories ayant un support inférieur à 50 échantillons.: Il est snon pertinent d'entraîner un modèle sur des classes avec si peu d'exemples. 
*  Les features textuelles :  (`product_name_decli`, `summary`, `description`,`brand` :  'brand' un bien remplit dans le dataset et peut aider à identifier certaines catégories.

### 2. Choix des Modèles

* **Approche 1 : Baseline Classique (TF-IDF + Régression Logistique)**
    * **Choix :** Un pipeline `scikit-learn` avec un pré-traitement de texte avancé (lemmatisation via `spaCy`) et un classifieur linéaire.
    * **Justification :** Permet d'établir une performance de référence solide de manière rapide et peu coûteuse. C'est un excellent candidat pour une mise en production rapide où la vitesse et la simplicité sont prioritaires.
* **Approche 2 : Deep Learning (Fine-tuning de CamemBERT)**
    * **Choix :** Fine-tuning d'un modèle Transformer pré-entraîné, spécialisé pour le français.
    * **Justification :** Vise la performance de pointe en exploitant la capacité du modèle à comprendre le contexte et la sémantique du texte, ce qui est particulièrement utile pour les cas ambigus et les catégories rares.

### 3. Évaluation et Comparaison

* **Méthodologie :** Un jeu d'évaluation de 200 produits a été créé via un **échantillonnage stratifié** pour une comparaison équitable des deux modèles.
* **Résultats :**
    * Le **modèle classique** a atteint une performance remarquable de **98% de précision**, mais a montré des faiblesses sur les classes les plus rares.
    * Le **modèle Transformer**, ré-entraîné avec la `brand`, a atteint **99% de précision** et a prouvé sa supériorité en classant correctement des catégories où le modèle classique échouait.
* **Conclusion :** Bien que le modèle Transformer soit techniquement supérieur, le modèle classique représente une alternative très compétitive, offrant un excellent compromis entre performance et coût. Le choix final dépendrait des contraintes du projet (besoin de précision maximale vs. besoin de rapidité et de simplicité).

---

## 🔮 Pistes d'Amélioration Futures

* **Analyse d'Erreurs :** Utiliser la matrice de confusion pour analyser en profondeur les erreurs du Transformer et comprendre les confusions résiduelles.
* **Hyperparameter Tuning :** Utiliser des librairies comme `Optuna` ou `Ray Tune` pour optimiser les hyperparamètres des deux modèles et potentiellement gagner les derniers points de performance.
* **Cross-Validation :** Pour le modèle classique (qui est rapide), mettre en place une validation croisée (k-fold) pour obtenir une estimation encore plus robuste de sa performance.
* **Déploiement à l'échelle :** Pour une application à fort trafic, déployer le modèle optimisé (`.onnx`) via un serveur d'inférence dédié (ex: NVIDIA Triton) pour maximiser le débit.

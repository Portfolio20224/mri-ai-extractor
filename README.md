**# 🧠 Multilingual Medical Data Extractor (LLM Weak Supervision)**

Extraction d'entités cliniques complexes (12 pathologies) à partir de comptes-rendus d'IRM non structurés et multilingues.

Ce projet illustre la conception d'un pipeline  de **Weak Supervision** (génération d'un *Silver Standard*) en utilisant les Modèles de Langage (LLMs) couplés à une architecture de production résiliente.

---

## 🎯 Alignement Métier

Ce repository démontre une expertise directe sur les activités clés de l'ingénierie Data/IA :

* **Développement de composants applicatifs Python :** Architecture Orientée Objet (POO) stricte séparant la logique métier (`annotator.py`), l'orchestration des flux de données (`pipeline.py`) et l'évaluation statistique (`evaluator.py`).
* **Optimisation de Prompts & IA :** Le système de prompting a été itérativement optimisé via l'analyse des Faux Positifs/Négatifs. Les frontières sémantiques complexes (ex: différencier un *épanchement physiologique* d'un *épanchement pathologique*, ou rattacher la *maladie de Hoffa* à la *synovite*) sont finement gérées.
* **Sensibilisation à la Production & Disponibilité :**
* Implémentation d'un **Throttling proactif** et d'une interception dynamique des erreurs API (HTTP 429 Quota Exhausted) pour un traitement continu sans interruption.
* Mécanisme de **Checkpointing** garantissant la reprise sur erreur pour les traitements massifs (Batch processing de > 4000 rapports).


* **Analyse et Test (Recette) :** Module d'évaluation robuste (`scikit-learn`) comparant les inférences de l'IA à un *Gold Standard* humain, générant matrices de confusion et métriques de classification.
* **Développement sécurisé & Bonnes pratiques :** Gestion des dépendances déterministe via **Poetry**, sécurisation des clés d'API via fichiers `.env` ignorés par Git.

---

## 🏗️ Architecture du Projet

```text
mri-ai-extractor/
├── data/                   # (Ignoré) Datasets d'entrée (Gold & Train)
├── out/                    # (Ignoré) Prédictions CSV
│   ├── confusion_matrices.png  # Généré par l'évaluateur
│   └── performance_summary.png # Généré par l'évaluateur
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration centralisée (Prompt, Labels)
│   ├── annotator.py        # Composant LLM (Inférence, Parsing JSON, Quotas)
│   ├── pipeline.py         # Orchestrateur (Throttling, Checkpointing)
│   └── evaluator.py        # Moteur de métriques (Scikit-Learn, Dataviz)
├── main.py                 # Point d'entrée d'exécution
├── pyproject.toml          # Dépendances Poetry
├── poetry.lock
└── README.md

```

---

## 🚀 Méthodologie du Pipeline

1. **Phase de Test (Gold Standard) :** Validation des invites (prompts) sur un échantillon certifié par des experts médicaux. Les métriques générées permettent d'ajuster les règles d'extraction sémantiques.
2. **Annotation Massive (Inference) :** Déploiement sécurisé sur des milliers de rapports bruts. Le pipeline résiste aux coupures réseau et gère l'espacement des requêtes.
3. **Filtre de Confiance (Silver Standard) :** L'IA auto-évalue la clarté du rapport (`confidence_rate`). Les rapports scorés à >90% forment un jeu de données structuré prêt à entraîner de futurs modèles NLP plus légers (ex: ClinicalBERT) via Distillation de Connaissances.

---

## 📊 Résultats et Performances

Le pipeline a été validé sur un Gold Standard avec d'excellentes métriques de classification binaire, démontrant la pertinence des règles sémantiques implémentées.

* **Taux de succès d'extraction JSON :** 100% (4407/4407 rapports annotés sans échec de parsing)
* **Confiance LLM moyenne :** 97.1%
* **Précision Globale (Accuracy) :** 85.7%
* **F1-Score Macro :** 84.3%

### Métriques clés (Échantillon Validation)

| Pathologie | F1-Score | Précision | Rappel |
| --- | --- | --- | --- |
| **ACL (Croisé Antérieur)** | 0.92 | 0.88 | 0.95 |
| **Medial Meniscus** | 0.83 | 0.79 | 0.88 |
| **Medial OA (Arthrose interne)** | 0.86 | 0.86 | 0.86 |
| **PF OA (Arthrose fémoro-patellaire)** | 0.76 | 1.00 | 0.61 |

*(Les matrices de confusion détaillées sont générées automatiquement dans le dossier `out/` lors de l'exécution).*
[Matrices de confusions](out/confusion_matrices.png)

### 🌟 Au-delà des métriques : L'IA comme outil d'audit du Gold Standard

L'une des découvertes majeures de ce projet a été la capacité de l'IA à surpasser la labellisation humaine sur la détection des faux signaux. Lors de l'analyse d'erreurs, 4 "Faux Négatifs" persistants sur la classe **PF OA** (Arthrose fémoro-patellaire) ont été isolés.

L'investigation des rapports textuels bruts a révélé que **l'IA avait raison de prédire 0**, et que le *Gold Standard* humain comportait des erreurs de labellisation manifestes :

* **Rapport Turc :** *"Distal kuadriseps ve patellar tendonlar normaldir. Patella normal yerleşimlidir."* (Traduction : Patella normale). Le Gold Standard affichait 1, l'IA a corrigé en 0.
* **Rapport Anglais :** *"OA of medial compartment"*. Aucune mention de la rotule dans le texte. L'IA a refusé d'halluciner une pathologie absente.
* **Rapport Allemand :** Aucune mention de la rotule ou de la trochlée dans l'intégralité du texte.
* **Rapport Bulgare :** *"Патела, хрущял на пателата... б.о."* (Traduction : Patella, cartilage patellaire... sans particularité).

**Impact métier :** Le pipeline LLM ne se contente pas de reproduire les annotations à grande échelle ; il agit comme un filtre de qualité strict, capable d'auditer, de nettoyer et de redresser les erreurs humaines présentes dans les bases de données cliniques de référence.

---

## ⚙️ Installation & Usage

Ce projet utilise [Poetry](https://www.google.com/search?q=https://python-poetry.org/) pour une gestion rigoureuse des dépendances et des environnements virtuels.

### 1. Cloner le repository

```bash
git clone https://github.com/votre-nom/mri-ai-extractor.git
cd mri-ai-extractor

```

### 2. Installer les dépendances

```bash
poetry install

```

### 3. Configurer l'environnement sécurisé

Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Google Gemini :

```env
GOOGLE_API_KEY="votre_clé_api_ici"

```

### 4. Lancer le pipeline

```bash
poetry run python main.py

```

## 💻 Tech Stack

* **Langages & Outils :** Python 3.10+, Pandas, NumPy, Poetry
* **IA & LLM :** LangChain, API Google Gemini Flash-Lite
* **Évaluation & Dataviz :** Scikit-Learn, Matplotlib (Backend Agg), Seaborn
# LSTM_Chem - Générateur et Analyseur de Molécules

## Description
LSTM_Chem est une application web basée sur Streamlit qui utilise un réseau de neurones LSTM pour générer et analyser des molécules.

## Installation

### Prérequis
- Python 3.7
- 8GB RAM minimum
- CUDA 11.2+ (optionnel, pour GPU)
- Anaconda ou Miniconda

### Configuration de l'environnement

1. Copier le répertoire et aller dedans :

```bash
cd LSTM_Chem-master_G
```

2. Créer l'environnement Conda avec le fichier environment.yml :

```bash
conda env create -f environment.yml
```

3. Activer l'environnement :

```bash
conda activate lstm_chem2
```

4. Mettre à jour Streamlit :

```bash
pip install streamlit --upgrade
```

### Lancer l'application
Toujours dans le dossier **LSTM_Chem-master_G** et en ayant activé l'environement conda **lstm_chem2**lancer la commande suivante à partir d'un terminal : 

```bash
streamlit run LSTM_interface.py
```

## Utilisation

### Fichiers d'entrée requis
- Fichier SMILES pour fine-tuning (.smi)  (nettoyé au préalable avec cleanup_smiles.py )
- Fichier SMILES de molécules connues pour l'analyse comparative (.smi) (si on a envie de comparer les moélcules générées a des molécules connues (facultatif))

### Fonctionnalités
- Génération de nouvelles molécules autour d'un groupe de molécules (fine tune)
- Visualisation des structures moléculaires générées 
- Analyse par PCA et descripteurs moléculaires
- Export des résultats

## Résolution des problèmes courants

Si vous rencontrez des erreurs avec protobuf :

```bash
pip uninstall protobuf
pip install protobuf==3.20.0
```

Pour désactiver CUDA si nécessaire, ajoutez au début du script :
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```



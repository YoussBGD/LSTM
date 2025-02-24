# generation.py
import streamlit as st
import os
import json
from lstm_chem.utils.config import process_config
from lstm_chem.model import LSTMChem
from lstm_chem.data_loader import DataLoader
from lstm_chem.finetuner import LSTMChemFinetuner
from src.config import FULL_CONFIG, BASE_CONFIG

def save_config_files(config_data):
    """Sauvegarde les fichiers de configuration."""
    # Mettre à jour la configuration complète
    full_config = FULL_CONFIG.copy()
    full_config.update({
        "sampling_temp": config_data["sampling_temp"],
        "smiles_max_length": config_data["smiles_max_length"],
        "finetune_epochs": config_data["finetune_epochs"],
        "finetune_batch_size": config_data["finetune_batch_size"],
        "finetune_data_filename": config_data["finetune_data_filename"]
    })
    
    # Sauvegarder config.json
    os.makedirs("experiments/2020-03-24/LSTM_Chem", exist_ok=True)
    with open("experiments/2020-03-24/LSTM_Chem/config.json", 'w') as f:
        json.dump(full_config, f, indent=2)
    
    # Mettre à jour et sauvegarder base_config.json
    base_config = BASE_CONFIG.copy()
    base_config.update({
        "sampling_temp": config_data["sampling_temp"],
        "smiles_max_length": config_data["smiles_max_length"],
        "finetune_epochs": config_data["finetune_epochs"],
        "finetune_batch_size": config_data["finetune_batch_size"],
        "finetune_data_filename": config_data["finetune_data_filename"]
    })
    
    with open("base_config.json", 'w') as f:
        json.dump(base_config, f, indent='\t')

def validate_config(config):
    """Valide la configuration."""
    if config["sampling_temp"] < 0 or config["sampling_temp"] > 1:
        raise ValueError("La température d'échantillonnage doit être entre 0 et 1")
    if config["smiles_max_length"] < 1:
        raise ValueError("La longueur maximale SMILES doit être positive")
    if config["finetune_epochs"] < 1:
        raise ValueError("Le nombre d'époques doit être positif")
    if config["finetune_batch_size"] < 1:
        raise ValueError("La taille du batch doit être positive")
    return True

def setup_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    directories = [
        "datasets",
        "experiments/2020-03-24/LSTM_Chem",
        "experiments/2020-03-24/LSTM_Chem/logs",
        "experiments/2020-03-24/LSTM_Chem/checkpoints"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def render_generation_page():
    """Rendu de la page de génération."""
    st.header("💻 Génération de Molécules")
    
    setup_directories()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration du Modèle")
        config = {
            "sampling_temp": st.slider(
                "Température d'échantillonnage",
                0.0, 1.0, 0.75, 0.01,
                help="Contrôle la créativité du modèle : proche de 0 = molécules similaires, proche de 1 = molécules plus originales"
            ),
            "smiles_max_length": st.number_input(
                "Longueur maximale SMILES",
                1, 200, 128,
                help="Longueur maximale des chaînes SMILES générées"
            ),
            "finetune_epochs": st.number_input(
                "Nombre d'époques",
                1, 100, 12,
                help="Nombre d'itérations d'entraînement"
            ),
            "finetune_batch_size": st.number_input(
                "Taille du batch",
                1, 32, 1,
                help="Nombre d'échantillons traités simultanément"
            ),
            "finetune_data_filename": FULL_CONFIG["finetune_data_filename"]
        }
    
    with col2:
        st.subheader("Données d'Entrée")
        finetune_file = st.file_uploader(
            "Fichier SMILES pour fine-tuning",
            type=['smi'],
            help="Fichier contenant les molécules d'entraînement au format SMILES"
        )
        
        if finetune_file:
            file_path = os.path.join("datasets", finetune_file.name)
            with open(file_path, "wb") as f:
                f.write(finetune_file.getbuffer())
            config["finetune_data_filename"] = f"./datasets/{finetune_file.name}"
            st.success(f"Fichier de fine-tuning sauvegardé: {finetune_file.name}")
            
        num_molecules = st.number_input(
            "Nombre de molécules à générer",
            min_value=1,
            value=100,
            help="Nombre total de nouvelles molécules à générer"
        )
        
        if st.button("🚀 Générer les molécules"):
            try:
                # Validation de la configuration
                validate_config(config)
                
                # Création de la barre de progression et du spinner
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                with st.spinner(""):  # Spinner vide pour ne pas dupliquer le texte
                    # Sauvegarde des configurations
                    progress_text.text("Initialisation du processus...")
                    progress_bar.progress(10)
                    save_config_files(config)
                    
                    # Initialisation et génération
                    progress_text.text("Préparation du modèle...")
                    progress_bar.progress(20)
                    config = process_config('experiments/2020-03-24/LSTM_Chem/config.json')
                    modeler = LSTMChem(config, session='finetune')
                    finetune_dl = DataLoader(config, data_type='finetune')
                    finetuner = LSTMChemFinetuner(modeler, finetune_dl)
                    
                    # Fine-tuning et génération
                    progress_text.text("Génération des molécules en cours...")
                    progress_bar.progress(40)
                    finetuner.finetune()
                    progress_bar.progress(70)
                    generated_smiles = finetuner.sample(num=num_molecules)
                    
                    # Sauvegarde des résultats
                    progress_bar.progress(90)
                    st.session_state['generated_smiles'] = generated_smiles
                    with open("generated_molecules.smi", "w") as f:
                        for smile in generated_smiles:
                            f.write(f"{smile}\n")
                    
                    # Finalisation
                    progress_bar.progress(100)
                    progress_text.text("Génération terminée!")
                
                # Bouton de téléchargement
                with open("generated_molecules.smi", "r") as f:
                    st.download_button(
                        "📥 Télécharger les molécules générées",
                        f.read(),
                        "generated_molecules.smi",
                        "text/plain",
                        help="Télécharger les molécules générées au format SMILES"
                    )
                        
            except Exception as e:
                st.error(f"Erreur lors de la génération: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    st.set_page_config(page_title="Génération de Molécules", layout="wide")
    render_generation_page()

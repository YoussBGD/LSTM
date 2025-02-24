# analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from rdkit import Chem
from sklearn.decomposition import PCA
import zipfile
import io
import os
from ..utils import calculate_descriptors, calculate_fingerprints

def plot_pca(known_fps, finetune_fps, generated_fps, fp_type, dimensions=2):
    """Réalise et affiche la PCA."""
    # Concaténation des fingerprints
    all_fps = np.vstack([known_fps, finetune_fps, generated_fps])
    
    # PCA
    n_components = 3 if dimensions == 3 else 2
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(all_fps)
    
    if dimensions == 3:
        # Création du graphique 3D interactif avec Plotly
        fig = go.Figure()
        
        # Ajout des points pour chaque catégorie
        fig.add_trace(go.Scatter3d(
            x=X[:len(known_fps), 0],
            y=X[:len(known_fps), 1],
            z=X[:len(known_fps), 2],
            mode='markers',
            name='Molécules connues',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.6
            )
        ))
        
        fig.add_trace(go.Scatter3d(
            x=X[len(known_fps):len(known_fps)+len(finetune_fps), 0],
            y=X[len(known_fps):len(known_fps)+len(finetune_fps), 1],
            z=X[len(known_fps):len(known_fps)+len(finetune_fps), 2],
            mode='markers',
            name='Molécules de fine tunning',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=0.8
            )
        ))
        
        fig.add_trace(go.Scatter3d(
            x=X[len(known_fps)+len(finetune_fps):, 0],
            y=X[len(known_fps)+len(finetune_fps):, 1],
            z=X[len(known_fps)+len(finetune_fps):, 2],
            mode='markers',
            name='Molécules générées',
            marker=dict(
                size=5,
                color='green',
                opacity=0.6
            )
        ))
        
        # Mise à jour du layout
        fig.update_layout(
            title=f'Analyse en Composantes Principales (Fingerprints {fp_type})',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            ),
            width=800,
            height=600
        )
        
        return fig, None  # Return None pour matplotlib figure
        
    else:
        # Version 2D avec Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.scatter(X[:len(known_fps), 0], X[:len(known_fps), 1],
                  c='blue', label='Molécules connues', alpha=0.6)
        ax.scatter(X[len(known_fps):len(known_fps)+len(finetune_fps), 0],
                  X[len(known_fps):len(known_fps)+len(finetune_fps), 1],
                  c='red', marker='*', s=200, label='Molécules de fine tunning')
        ax.scatter(X[len(known_fps)+len(finetune_fps):, 0],
                  X[len(known_fps)+len(finetune_fps):, 1],
                  c='green', label='Molécules générées', alpha=0.6)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title(f'Analyse en Composantes Principales\n(Fingerprints {fp_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return None, fig  # Return None pour plotly figure

def create_distribution_plots(known_df, finetune_df, generated_df):
    """Crée les graphiques de distribution pour chaque descripteur."""
    figures = {}
    
    for col in known_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.kdeplot(data=known_df[col], label="Connues", ax=ax)
        sns.kdeplot(data=finetune_df[col], label="Fine-tune", ax=ax)
        sns.kdeplot(data=generated_df[col], label="Générées", ax=ax)
        
        plt.title(f'Distribution de {col}')
        plt.xlabel(col)
        plt.ylabel('Densité')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        figures[f'distribution_{col}'] = fig
        
    return figures

def save_analysis_results(known_df, finetune_df, generated_df, figures):
    """Sauvegarde les résultats d'analyse dans un fichier ZIP."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Sauvegarder les DataFrames
        for name, df in [("known", known_df), ("finetune", finetune_df), ("generated", generated_df)]:
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{name}_descriptors.csv", csv_data)
        
        # Sauvegarder les statistiques descriptives
        stats_content = []
        for name, df in [("Molécules connues", known_df),
                        ("Molécules de fine-tuning", finetune_df),
                        ("Molécules générées", generated_df)]:
            stats_content.extend([
                f"\nStatistiques des {name}:",
                df.describe().to_string(),
                "\n" + "="*50 + "\n"
            ])
        
        zip_file.writestr("statistical_summary.txt", "\n".join(stats_content))
        
        # Sauvegarder les figures
        for name, fig in figures.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            zip_file.writestr(f"{name}.png", img_buffer.getvalue())
            plt.close(fig)
    
    return zip_buffer

def render_analysis_page():
    """Rendu de la page d'analyse."""
    st.header("📊 Analyse des Molécules")
    
    if 'generated_smiles' not in st.session_state:
        st.info("Veuillez d'abord générer des molécules dans l'onglet Génération")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        known_file = st.file_uploader(
            "Fichier SMILES de référence (molécules connues)",
            type=['smi'],
            help="Fichier contenant les molécules de référence au format SMILES"
        )
    
    with col2:
        finetune_file = st.file_uploader(
            "Fichier SMILES de fine-tuning",
            type=['smi'],
            help="Fichier contenant les molécules d'entraînement au format SMILES"
        )
    
    if not (known_file and finetune_file):
        st.warning("Veuillez charger les fichiers de molécules connues et de fine-tuning pour l'analyse")
        return
    
    known_smiles = [line.decode().strip() for line in known_file]
    finetune_smiles = [line.decode().strip() for line in finetune_file]
    generated_smiles = st.session_state['generated_smiles']
    
    known_mols = [Chem.MolFromSmiles(s) for s in known_smiles if Chem.MolFromSmiles(s)]
    finetune_mols = [Chem.MolFromSmiles(s) for s in finetune_smiles if Chem.MolFromSmiles(s)]
    generated_mols = [Chem.MolFromSmiles(s) for s in generated_smiles if Chem.MolFromSmiles(s)]
    
    st.subheader("Statistiques générales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Molécules connues", len(known_mols))
    with col2:
        st.metric("Molécules d'entraînement", len(finetune_mols))
    with col3:
        st.metric("Molécules générées", len(generated_mols))
    
    st.subheader("Configuration de l'analyse")
    col1, col2 = st.columns(2)
    
    with col1:
        fp_type = st.selectbox(
            "Type de Fingerprints pour PCA",
            ["MACCS", "Morgan"],
            help="MACCS: 167 bits prédéfinis\nMorgan: Fingerprints circulaires (ECFP4)"
        )
    
    with col2:
        pca_dimensions = st.radio(
            "Dimensions PCA",
            ["2D", "3D"],
            horizontal=True
        )
    
    with st.spinner("Calcul des descripteurs et fingerprints..."):
        known_fps = np.array([calculate_fingerprints(mol, fp_type) for mol in known_mols if mol is not None])
        finetune_fps = np.array([calculate_fingerprints(mol, fp_type) for mol in finetune_mols if mol is not None])
        generated_fps = np.array([calculate_fingerprints(mol, fp_type) for mol in generated_mols if mol is not None])
        
        known_desc = [calculate_descriptors(mol) for mol in known_mols if mol is not None]
        finetune_desc = [calculate_descriptors(mol) for mol in finetune_mols if mol is not None]
        generated_desc = [calculate_descriptors(mol) for mol in generated_mols if mol is not None]
        
        known_df = pd.DataFrame([d for d in known_desc if d is not None])
        finetune_df = pd.DataFrame([d for d in finetune_desc if d is not None])
        generated_df = pd.DataFrame([d for d in generated_desc if d is not None])
    
    # Export des résultats
    st.subheader("Export des Résultats")
    
    # Création des figures de distribution
    figures = create_distribution_plots(known_df, finetune_df, generated_df)
    
    # Création de la PCA
    plotly_fig, mpl_fig = plot_pca(known_fps, finetune_fps, generated_fps, fp_type, 
                                  dimensions=3 if pca_dimensions == "3D" else 2)
    
    if mpl_fig:
        figures['pca_plot'] = mpl_fig
    
    zip_buffer = save_analysis_results(known_df, finetune_df, generated_df, figures)
    
    st.download_button(
        label="📥 Télécharger les résultats d'analyse",
        data=zip_buffer.getvalue(),
        file_name="analyse_molecules.zip",
        mime="application/zip",
        help="Télécharger tous les résultats (graphiques, données et statistiques)"
    )
    
    # Affichage PCA
    st.subheader("Analyse en Composantes Principales")
    if plotly_fig:
        st.plotly_chart(plotly_fig, use_container_width=True)
    else:
        st.pyplot(mpl_fig)
    
    # Statistiques descriptives
    st.subheader("Analyse des Descripteurs Moléculaires")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Molécules Connues")
        st.dataframe(known_df.describe())
    
    with col2:
        st.write("Molécules Fine-tune")
        st.dataframe(finetune_df.describe())
    
    with col3:
        st.write("Molécules Générées")
        st.dataframe(generated_df.describe())
    
    # Affichage des distributions
    st.subheader("Distribution des Propriétés")
    for name, fig in figures.items():
        if name != 'pca_plot':
            st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="Analyse de Molécules", layout="wide")
    render_analysis_page()

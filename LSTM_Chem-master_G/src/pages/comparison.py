# comparison.py
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import math
import shutil
import os
from datetime import datetime
import plotly.express as px
from ..utils.molecular_utils import (
    calculate_descriptors, calculate_similarity, calculate_fingerprints,
    perform_statistical_analysis, perform_pca_analysis, get_available_similarity_methods,
    find_duplicate_groups, get_molecule_counts
)
from ..utils.structure_utils import save_2d_structures, generate_3d_structures

@st.cache_data
def process_molecules(finetune_smiles, generated_smiles):
    """Traite et met en cache les molécules de base."""
    finetune_mols = [Chem.MolFromSmiles(s) for s in finetune_smiles if Chem.MolFromSmiles(s)]
    generated_mols = [Chem.MolFromSmiles(s) for s in generated_smiles if Chem.MolFromSmiles(s)]
    return finetune_mols, generated_mols

def configure_similarity_settings():
    """Configure les paramètres de similarité."""
    st.subheader("🎯 Paramètres de similarité")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_methods = get_available_similarity_methods()
        similarity_method = st.selectbox(
            "Méthode de similarité",
            options=list(available_methods.keys()),
            help="Sélectionnez la méthode de calcul de similarité"
        )
        st.caption(available_methods[similarity_method]['description'])
        
        similarity_params = {}
        if similarity_method == 'Morgan':
            radius = st.slider(
                "Rayon ECFP",
                min_value=2,
                max_value=5,
                value=2,
                help="ECFP4=2, ECFP6=3, etc."
            )
            similarity_params['radius'] = radius
            
    with col2:
        similarity_threshold = st.slider(
            "Seuil de similarité minimum",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        sort_order = st.radio(
            "Tri des résultats (selon la similarité)",
            ["Décroissant", "Croissant"],
            horizontal=True
        )
    
    return similarity_method, similarity_params, similarity_threshold, sort_order

def configure_display_settings():
    """Configure les paramètres d'affichage."""
    st.subheader("🖥️ Options d'affichage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_molecules = st.number_input(
            "Nombre maximum de molécules à afficher",
            min_value=1,
            value=10,
            step=1
        )
    
    with col2:
        num_cols = st.slider(
            "Nombre de molécules par ligne",
            min_value=1,
            max_value=5,
            value=3
        )
        
    with col3:
        show_duplicates = st.checkbox(
            "Inclure les molécules dupliquées",
            value=False,
            help="Si désactivé, seule une instance de chaque molécule sera affichée"
        )
    
    return max_molecules, num_cols, show_duplicates

def configure_analysis_settings():
    """Configure les paramètres d'analyse."""
    st.subheader("📊 Options d'analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_stats = st.checkbox("Afficher les analyses statistiques", value=False)
    with col2:
        show_pca = st.checkbox("Afficher l'analyse PCA", value=False)
    
    return show_stats, show_pca

def calculate_all_similarities(reference_mol, generated_mols, method='Morgan', **kwargs):
    """Calcule les similarités."""
    similarities = []
    for i, mol in enumerate(generated_mols):
        similarity = calculate_similarity(reference_mol, mol, method=method, **kwargs)
        smiles = Chem.MolToSmiles(mol)
        original_index = i + 1
        similarities.append((similarity, i, smiles, mol, original_index))
    return similarities

def filter_similar_molecules(all_similarities, threshold, max_molecules=None, sort_order="Décroissant", show_duplicates=False):
    """Filtre les molécules selon le seuil de similarité et les trie."""
    filtered_mols = [item for item in all_similarities if item[0] > threshold]
    
    # Détection des doublons
    duplicate_groups = find_duplicate_groups(filtered_mols)
    
    # Si on ne veut pas afficher les doublons, on garde seulement une instance de chaque molécule
    if not show_duplicates:
        unique_mols = []
        used_smiles = set()
        for mol in filtered_mols:
            smiles = Chem.MolToSmiles(mol[3], canonical=True)
            if smiles not in used_smiles:
                unique_mols.append(mol)
                used_smiles.add(smiles)
        filtered_mols = unique_mols
    
    # Tri des molécules
    filtered_mols.sort(key=lambda x: x[0], reverse=(sort_order == "Décroissant"))
    
    total_similar = len(filtered_mols)
    display_mols = filtered_mols[:max_molecules] if max_molecules else filtered_mols
    
    return display_mols, filtered_mols, total_similar, duplicate_groups

def create_similarity_dataframe(reference_name, reference_smiles, similar_mols, method, **kwargs):
    """Crée un DataFrame avec les informations de similarité."""
    method_info = f"{method}"
    if 'radius' in kwargs:
        method_info += f" (rayon={kwargs['radius']})"
    
    data = {
        'Nom': [reference_name] + [f'Molécule générée {mol[4]}' for mol in similar_mols],
        'SMILES': [reference_smiles] + [smiles for _, _, smiles, _, _ in similar_mols],
        'Similarité': [1.0] + [sim for sim, _, _, _, _ in similar_mols],
        'Méthode': [method_info] * (len(similar_mols) + 1)
    }
    
    return pd.DataFrame(data)

def display_molecule_grid(reference_mol, reference_smiles, reference_idx, molecules, similarity_method, num_cols=3):
    """Affiche une grille de molécules avec défilement."""
    st.subheader("Molécule clé")
    img_ref = Draw.MolToImage(reference_mol)
    st.image(img_ref)
    st.write(f"Molécule clé {reference_idx + 1}")
    st.write(f"SMILES: ```{reference_smiles}```")
    
    st.subheader(f"Molécules similaires (Méthode: {similarity_method})")
    num_mols = len(molecules)
    num_rows = math.ceil(num_mols / num_cols)
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < num_mols:
                similarity, _, smiles, mol, original_index = molecules[idx]
                with cols[col]:
                    img = Draw.MolToImage(mol)
                    st.image(img)
                    st.write(f"Molécule générée {original_index}")
                    st.write(f"Similarité: {similarity:.3f}")
                    st.write(f"SMILES: ```{smiles}```")

def display_duplicate_info(duplicate_groups):
    """Affiche les informations sur les doublons avec visualisation des molécules."""
    if len(duplicate_groups) > 0:
        st.warning(f"⚠️ {len(duplicate_groups)} groupe(s) de molécules identiques détectés")
        with st.expander("Voir les détails des doublons"):
            # Sélection du groupe à visualiser
            group_idx = st.selectbox(
                "Sélectionner un groupe de doublons à visualiser",
                range(len(duplicate_groups)),
                format_func=lambda x: f"Groupe {x+1} ({len(duplicate_groups[x])} molécules)"
            )
            
            # Affichage des SMILES du groupe sélectionné
            st.write(f"**SMILES des molécules du groupe {group_idx + 1}:**")
            for _, _, smiles, _, idx in duplicate_groups[group_idx]:
                st.write(f"- Molécule générée {idx}: ```{smiles}```")
            
            # Visualisation des molécules du groupe
            st.write("**Visualisation des molécules:**")
            cols = st.columns(min(3, len(duplicate_groups[group_idx])))
            for i, (_, _, _, mol, idx) in enumerate(duplicate_groups[group_idx]):
                with cols[i % 3]:
                    img = Draw.MolToImage(mol)
                    st.image(img)
                    st.write(f"Molécule générée {idx}")

def display_statistical_analysis(stats_results):
    """Affiche les résultats des analyses statistiques."""
    st.subheader("Analyses statistiques")
    
    stats_df = pd.DataFrame({
        'Descripteur': [],
        'Moyenne référence': [],
        'Moyenne générées': [],
        'Écart-type générées': [],
        'Statistique t': [],
        'p-value': []
    })
    
    for desc, results in stats_results.items():
        stats_df = pd.concat([stats_df, pd.DataFrame({
            'Descripteur': [desc],
            'Moyenne référence': [results['reference_mean']],
            'Moyenne générées': [results['generated_mean']],
            'Écart-type générées': [results['generated_std']],
            'Statistique t': [results['t_statistic']],
            'p-value': [results['p_value']]
        })], ignore_index=True)
    
    st.dataframe(stats_df)

def display_pca_analysis(pca_results, reference_name="Molécule clé"):
    """Affiche les résultats de l'analyse PCA."""
    st.subheader("Analyse en Composantes Principales")
    
    ref_coords = pca_results['reference']
    gen_coords = pca_results['generated']
    
    df_pca = pd.DataFrame({
        'PC1': [ref_coords[0]] + list(gen_coords[:, 0]),
        'PC2': [ref_coords[1]] + list(gen_coords[:, 1]),
        'Type': [reference_name] + ['Molécule générée'] * len(gen_coords)
    })
    
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Type',
                    title="Analyse PCA des fingerprints moléculaires")
    st.plotly_chart(fig)
    
    st.write(f"Variance expliquée: {pca_results['explained_variance'][0]:.2%} (PC1), "
             f"{pca_results['explained_variance'][1]:.2%} (PC2)")

def render_comparison_page():
    """Rendu de la page de comparaison."""
    st.header("🔄 Recherche similarité (Molécules générées vs autres bases de données)")
    
    if 'generated_smiles' not in st.session_state:
        st.info("Veuillez d'abord générer des molécules dans l'onglet Génération")
        return
    
    finetune_file = st.file_uploader(
        "Exemple : Fichier SMILES de fine-tuning",
        type=['smi'],
        help="Fichier contenant les molécules d'entraînement"
    )
    
    if not finetune_file:
        st.warning("Veuillez charger le fichier de molécules de fine-tuning")
        return

    # Configuration des paramètres
    with st.expander("⚙️ Configuration", expanded=True):
        similarity_method, similarity_params, similarity_threshold, sort_order = configure_similarity_settings()
        max_molecules, num_cols, show_duplicates = configure_display_settings()
        #show_stats, show_pca = configure_analysis_settings()

    # Traitement des molécules
    finetune_smiles = [line.decode().strip() for line in finetune_file]
    generated_smiles = st.session_state['generated_smiles']

    with st.spinner("Traitement initial des molécules..."):
        finetune_mols, generated_mols = process_molecules(
            finetune_smiles, generated_smiles
        )
    
    # Sélection de la molécule de référence
    st.subheader("Sélection de la molécule de référence")
    selected_idx = st.selectbox(
        "Molécule de référence",
        range(len(finetune_mols)),
        format_func=lambda x: f"Molécule clé {x+1}"
    )
    
    reference_mol = finetune_mols[selected_idx]
    reference_smiles = Chem.MolToSmiles(reference_mol)

    # Calcul des similarités et filtrage
    with st.spinner("Calcul des similarités..."):
        all_similarities = calculate_all_similarities(
            reference_mol, 
            generated_mols,
            method=similarity_method,
            **similarity_params
        )
        display_mols, all_filtered_mols, total_similar, duplicate_groups = filter_similar_molecules(
            all_similarities, 
            similarity_threshold,
            max_molecules,
            sort_order,
            show_duplicates
        )
    
    # Affichage des informations sur les doublons
    display_duplicate_info(duplicate_groups)
    
    # Affichage du nombre de molécules
    total_avec_doublons, total_sans_doublons = get_molecule_counts(all_filtered_mols, duplicate_groups)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Molécules similaires trouvées :")
    with col2:
        st.write(f"**Total avec doublons** : {total_avec_doublons}")
        st.write(f"**Total sans doublons** : {total_sans_doublons}")

    if max_molecules and total_similar > max_molecules:
        st.write(f"Affichage limité aux {max_molecules} molécules les plus similaires")
    
    if total_similar > 0:
        # Affichage des molécules
        display_molecule_grid(
            reference_mol, 
            reference_smiles, 
            selected_idx, 
            display_mols, 
            similarity_method,
            num_cols
        )
        
        # Création et affichage du DataFrame
        reference_name = f"Molécule clé {selected_idx + 1}"
        df_similar = create_similarity_dataframe(
            reference_name, 
            reference_smiles, 
            display_mols,
            similarity_method,
            **similarity_params
        )
        
        st.subheader("Tableau des similarités")
        st.dataframe(df_similar)
        
        # Export CSV
        st.download_button(
            "📥 Télécharger les données de similarité (CSV)",
            df_similar.to_csv(index=False),
            "molecules_similaires.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Analyses statistiques et PCA
        #if show_stats or show_pca:
        #    st.markdown("---")
        #    st.subheader("Analyses")
            
        #    if show_stats:
        #        with st.spinner("Calcul des analyses statistiques..."):
        #            reference_desc = calculate_descriptors(reference_mol)
        #            generated_desc = [calculate_descriptors(mol) for _, _, _, mol, _ in all_filtered_mols]
        #            stats_results = perform_statistical_analysis(reference_desc, generated_desc)
        #            display_statistical_analysis(stats_results)
            
        #    if show_pca:
        #        with st.spinner("Calcul de l'analyse PCA..."):
        #            reference_fp = calculate_fingerprints(reference_mol)
        #            generated_fps = [calculate_fingerprints(mol) for _, _, _, mol, _ in all_filtered_mols]
        #            pca_results = perform_pca_analysis(reference_fp, generated_fps)
        #            display_pca_analysis(pca_results)
        
        
        # Options d'export des structures
        st.markdown("---")
        st.subheader("Export des Structures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Options de génération 3D:")
            add_h = st.checkbox("Ajouter les hydrogènes", value=True)
            if add_h:
                ph_value = st.number_input("pH", value=7.4, min_value=0.0, max_value=14.0, step=0.1)
        
        if st.button("🔄 Générer les structures 2D/3D"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Génération des structures en cours..."):
                    base_name = f"molecule_{selected_idx + 1}_sim{similarity_threshold}"
                    output_dir = os.path.join("generated_structures", base_name)
                    
                    status_text.text("Préparation des molécules...")
                    progress_bar.progress(10)
                    
                    # Utiliser les molécules selon l'option des doublons
                    if not show_duplicates:
                        mols_to_export = [reference_mol] + [mol for _, _, _, mol, _ in display_mols]
                    else:
                        mols_to_export = [reference_mol] + [mol for _, _, _, mol, _ in all_filtered_mols]
                    
                    status_text.text(f"Génération des structures 2D...")
                    progress_bar.progress(20)
                    
                    sdf_2d = save_2d_structures(
                        mols_to_export,
                        output_dir,
                        selected_idx + 1,
                        similarity_threshold
                    )
                    
                    status_text.text(f"Génération des structures 3D...")
                    progress_bar.progress(50)
                    sdf_3d = generate_3d_structures(
                        sdf_2d,
                        output_dir,
                        selected_idx + 1,
                        similarity_threshold,
                        add_hydrogens=add_h,
                        ph=ph_value if add_h else None
                    )
                    
                    status_text.text("Création de l'archive ZIP...")
                    progress_bar.progress(80)
                    zip_file = os.path.join(output_dir, "structures.zip")
                    shutil.make_archive(
                        os.path.join(output_dir, "structures"),
                        'zip',
                        output_dir
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Génération terminée!")
                    
                    with open(zip_file, "rb") as f:
                        st.download_button(
                            "📥 Télécharger les structures (2D/3D)",
                            f,
                            file_name=f"{base_name}_structures.zip",
                            mime="application/zip"
                        )
                    
                    st.success(f"Structures générées avec succès! ({len(mols_to_export)} molécules)")
                    
            except Exception as e:
                st.error(f"Erreur lors de la génération des structures: {str(e)}")
                st.exception(e)
                
    else:
        st.warning(f"Aucune molécule trouvée avec une similarité supérieure à {similarity_threshold}")

if __name__ == "__main__":
    st.set_page_config(page_title="Comparaison de Molécules", layout="wide")
    render_comparison_page()

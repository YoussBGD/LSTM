# visualization.py
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import math

def create_molecule_grid(mols, cols_per_row, subimg_size=(300, 300)):
    """Crée une grille d'images de molécules."""
    if not mols:
        return None
        
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=cols_per_row,
        subImgSize=subimg_size,
        legends=[f"Mol_{i+1}" for i in range(len(mols))]
    )

def render_visualization_page():
    """Rendu de la page de visualisation."""
    st.header("🔍 Visualisation des Molécules")
    
    if 'generated_smiles' not in st.session_state:
        st.info("Veuillez d'abord générer des molécules dans l'onglet Génération")
        return
        
    # Conversion des SMILES en molécules
    mols = [Chem.MolFromSmiles(smile) for smile in st.session_state['generated_smiles']]
    mols = [mol for mol in mols if mol is not None]  # Filtrer les molécules invalides
    
    if not mols:
        st.warning("Aucune molécule valide à afficher")
        return
    
    # Options de visualisation
    col1, col2 = st.columns(2)
    
    with col1:
        cols_per_row = st.slider(
            "Nombre de molécules par ligne",
            2, 6, 4,
            help="Ajuster le nombre de molécules affichées par ligne"
        )
        
    with col2:
        show_count = st.slider(
            "Nombre de molécules à afficher",
            1, len(mols), min(20, len(mols)),
            help="Nombre total de molécules à afficher"
        )
    
    # Options supplémentaires
    with st.expander("Options avancées"):
        img_size = st.slider(
            "Taille des images (pixels)",
            100, 500, 300,
            help="Taille de chaque image de molécule"
        )
        show_index = st.checkbox(
            "Afficher les numéros",
            True,
            help="Afficher le numéro de chaque molécule"
        )
    
    # Affichage des molécules
    st.subheader(f"Visualisation des molécules (1-{show_count})")
    
    # Créer des lignes de molécules
    for i in range(0, show_count, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < show_count and i + j < len(mols):
                mol = mols[i + j]
                img = Draw.MolToImage(mol, size=(img_size, img_size))
                caption = f"Molécule {i+j+1}\n{Chem.MolToSmiles(mol)}" if show_index else Chem.MolToSmiles(mol)
                col.image(img, caption=caption, use_column_width=True)
    
    # Option de téléchargement de la grille complète
    if st.button("📥 Générer une grille d'images"):
        grid_img = create_molecule_grid(
            mols[:show_count],
            cols_per_row,
            (img_size, img_size)
        )
        if grid_img:
            st.image(grid_img)
            # Conversion de l'image pour le téléchargement
            from io import BytesIO
            buf = BytesIO()
            grid_img.save(buf, format="PNG")
            st.download_button(
                "💾 Télécharger la grille",
                buf.getvalue(),
                "molecule_grid.png",
                "image/png"
            )

if __name__ == "__main__":
    st.set_page_config(page_title="Visualisation de Molécules", layout="wide")
    render_visualization_page()

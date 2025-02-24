import streamlit as st
from src.pages import (
    render_generation_page,
    render_visualization_page,
    render_analysis_page,
    render_comparison_page
)

def main():
    st.set_page_config(
        page_title="LSTM_Chem",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 LSTM_Chem")
    
    # Création des onglets
    tabs = st.tabs([
        "💻 Génération",
        "🔍 Visualisation",
        "📊 Analyse",
        "🔄 Recherche de similarités"
    ])
    
    # Rendu de chaque onglet
    with tabs[0]:
        render_generation_page()
        
    with tabs[1]:
        render_visualization_page()
        
    with tabs[2]:
        render_analysis_page()
        
    with tabs[3]:
        render_comparison_page()

if __name__ == "__main__":
    main()

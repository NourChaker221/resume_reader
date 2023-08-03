import streamlit as st
from streamlit_option_menu import option_menu
from ParsingCv import ParsingCv
from classifier import classifier
from principale import principale
from similarity import similarity
def set_background_image(image_path):
    page_bg_img = f"""
        <style>
            body {{
                background-image: url("{image_path}");
                background-size: cover;
            }}
        </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
def acceuil():
    st.set_page_config(page_title="Resume Reader", page_icon="images/employee-resume-icon-free-vector.jpg", layout="wide")
    # Définir l'image de fond
    background_image = "images/2_720.jpg"
    set_background_image(background_image)
    # Insérer le logo en haut de la page et le rendre plus petit
    logo_path = "4.png"
    logo_width = 300  # Modifier cette valeur pour ajuster la largeur du logo
    st.image(logo_path, width=logo_width, use_column_width=False)
    # Personnaliser les couleurs du menu
    menu_bg_color = "#00F7FF"  # Couleur de fond du menu
    menu_text_color = "#FFFFFF"  # Couleur du texte du menu
    menu_border_color = "#007BFF"  # Couleur de la bordure du menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Classifier", "Parser ", "Similarity"],
        icons=['house', 'clipboard-check', "search", 'clipboard-data'],
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "bg_color": menu_bg_color,
            "text_color": menu_text_color,
            "border_color": menu_border_color
        }
    )
    if selected == "Home":
        print("acceuil")
        principale()
    elif selected == "Classifier":
        classifier()
    elif selected == "Parser ":
        ParsingCv()
    elif selected == "Similarity":
        similarity()
acceuil()
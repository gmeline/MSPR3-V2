import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Elixxion Dashboard",
    page_icon="📊",
    layout="wide"
)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data_clean", "table_finale_HDF_famille.csv")
df = pd.read_csv(DATA)

# ============================================================
# Menu à gauche
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

st.sidebar.title("Navigation")

if st.sidebar.button("Accueil"):
    st.session_state.page = "Accueil"

if st.sidebar.button("Visualisations"):
    st.session_state.page = "Visualisations"

if st.sidebar.button("Exploration"):
    st.session_state.page = "Exploration"

if st.sidebar.button("Machine Learning"):
    st.session_state.page = "Machine Learning"
page = st.session_state.page



# ============================================================
# Page d'accueil
# ============================================================
if page == "Accueil":
    st.title("Bienvenue sur l'application de Elexxion")
    st.write("Cette application présente les visualisations du projet MSPR.")

# ============================================================
#  Page Visualisation
# ============================================================
elif page == "Visualisations":
    import seaborn as sns

    st.title("Visualisations exploratoires")

    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("Visualisations exploratoires", fontsize=18)

    # ============================================================
    # Histogramme sur le taux d'abstention
    # ============================================================
    plt.subplot(2, 3, 1)
    if "% Abs/Ins" in df.columns:
        plt.hist(df["% Abs/Ins"], bins=20, color="#DD8452")
        plt.title("Abstention (%)")
    else:
        plt.text(0.5, 0.5, "Colonne '% Abs/Ins' manquante", ha="center")

    # ============================================================
    # Histogramme sur le taux de participation
    # ============================================================
    plt.subplot(2, 3, 2)
    if "% Vot/Ins" in df.columns:
        plt.hist(df["% Vot/Ins"], bins=20, color="#4C72B0")
        plt.title("Participation (%)")
    else:
        plt.text(0.5, 0.5, "Colonne '% Vot/Ins' manquante", ha="center")

    # ============================================================
    # Histogramme sur le taux de chomage
    # ============================================================
    plt.subplot(2, 3, 3)
    if "taux_chomage" in df.columns:
        plt.hist(df["taux_chomage"], bins=20, color="#55A868")
        plt.title("Taux de chômage")
    else:
        plt.text(0.5, 0.5, "Colonne 'taux_chomage' manquante", ha="center")

    
    # ============================================================
    # Heatmap de corrélation
    # ============================================================
    plt.subplot(2, 3, 4)
    cols = ["taux_chomage", "% Abs/Ins", "% Vot/Ins", "% Blancs/Ins"]
    cols = [c for c in cols if c in df.columns]
    if len(cols) >= 2:
        sns.heatmap(df[cols].corr(), cmap="coolwarm", annot=True)
        plt.title("Corrélations")
    else:
        plt.text(0.5, 0.5, "Colonnes insuffisantes", ha="center")

    # ============================================================
    # Scatter plot du chomage vs voix extreme droite
    # ============================================================
    plt.subplot(2, 3, 5)
    if {"taux_chomage", "voix_exp"}.issubset(df.columns):
        df_extreme = df[df["famille_politique"] == "extreme_droite"]
        plt.scatter(df_extreme["taux_chomage"], df_extreme["voix_exp"], color="#C44E52")
        plt.title("Chômage vs Voix extrême droite")
        plt.xlabel("Taux de chômage")
        plt.ylabel("Voix extrême droite")
    else:
        plt.text(0.5, 0.5, "Colonnes manquantes", ha="center")

    # ============================================================
    # Boxplot sur le chomage par famille politique
    # ============================================================
    plt.subplot(2, 3, 6)
    if {"famille_politique", "taux_chomage"}.issubset(df.columns):
        sns.boxplot(data=df, x="famille_politique", y="taux_chomage")
        plt.xticks(rotation=45)
        plt.title("Chômage par famille politique")
    else:
        plt.text(0.5, 0.5, "Colonnes manquantes", ha="center")

    st.pyplot(fig)

# ============================================================
# Page Exploration 
# ============================================================
elif page == "Exploration":
    st.title("Exploration par département")

    # ============================================================
    # Selection du département
    # ============================================================
    departements = sorted(df["departement"].unique())
    dep = st.selectbox("Choisir un département :", departements)
    df_dep = df[df["departement"] == dep]
    st.subheader(f"Département : {dep}")

    # ============================================================
    # Affichage des indicateurs clés en fonction du département
    # ============================================================
    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de chômage", f"{df_dep['taux_chomage'].mean():.2f} %")
    col2.metric("Abstention", f"{df_dep['% Abs/Ins'].mean():.2f} %")
    col3.metric("Participation", f"{df_dep['% Vot/Ins'].mean():.2f} %")

    # ============================================================
    # 1er graphique : voix par famille politique
    # ============================================================
    st.subheader("Répartition des voix par famille politique")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    df_dep.groupby("famille_politique")["voix_exp"].sum().plot(kind="bar", ax=ax1, color="#4C72B0")
    ax1.set_ylabel("Voix exprimées")
    st.pyplot(fig1)

    # ============================================================
    # 2eme graphique : histogramme des voix
    # ============================================================
    st.subheader("Distribution des voix exprimées")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(df_dep["voix_exp"], bins=15, color="#DD8452")
    ax2.set_xlabel("Voix exprimées")
    st.pyplot(fig2)

    # ============================================================
    # Tableau des candidats
    # ============================================================
    st.subheader("Détails par candidat")
    st.dataframe(df_dep[["nom", "prenom", "voix_exp", "% Voix/Ins", "famille_politique"]])

# ============================================================
# Page machine learning
# ============================================================
elif page == "Machine Learning":
    st.title("Visualisations Machine Learning")

    IMG_DIR = os.path.join(BASE, "images")
    ml_images = [
        ("Comparaison des scores", "ml_accuracy_comparison.png"),
        ("Matrices de confusion", "ml_confusion_matrices.png"),
        ("Importance des variables", "ml_feature_importance.png"),
        ("Courbes ROC", "ml_roc_curves.png")
    ]

    for title, filename in ml_images:
        path = os.path.join(IMG_DIR, filename)
        if os.path.exists(path):
            st.subheader(f"{title}")
            st.image(path, use_column_width=True)
        else:
            st.warning(f"Image manquante : {filename}")

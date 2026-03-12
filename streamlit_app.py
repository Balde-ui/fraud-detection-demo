import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# -- Configuration --
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)

# -- Chargement du modèle et des données --
@st.cache_resource
def load_model():
    with open("models/fraud_model_v2.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/sample_demo.csv")

model = load_model()
df = load_data()

# -- Feature Engineering (même logique que fraud_model_v2.py) --
def prepare_features(df):
    df = df.copy()
    df['Amount_log'] = np.log1p(df['Amount'])
    df['hour_of_day'] = (df['Time'] / 3600) % 24
    df = df.drop(columns=['Time', 'Amount', 'Class'], errors='ignore')
    return df

# -- Prédictions sur tout le dataset --
X = prepare_features(df)
df['fraud_probability'] = model.predict_proba(X)[:, 1]
df['is_fraud_predicted'] = model.predict(X)
df['risk_level'] = df['fraud_probability'].apply(
    lambda p: "🔴 HIGH" if p >= 0.7 else ("🟡 MEDIUM" if p >= 0.3 else "🟢 LOW")
)

# ── HEADER ──
st.title("🏦 Fraud Detection System")
st.markdown("Analyse automatique de transactions bancaires par Machine Learning.")
st.divider()

# ── KPIs ──
total = len(df)
fraudes = df['is_fraud_predicted'].sum()
legit = total - fraudes
montant_risque = df[df['is_fraud_predicted'] == 1]['Amount'].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Transactions analysées", total)
k2.metric("✅ Légitimes", legit)
k3.metric("🚨 Fraudes détectées", fraudes)
k4.metric("💰 Montant à risque", f"${montant_risque:,.2f}")

st.divider()

# ── GRAPHIQUES ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des scores de risque")
    hist_data = pd.cut(
        df['fraud_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW 🟢', 'MEDIUM 🟡', 'HIGH 🔴']
    ).value_counts()
    st.bar_chart(hist_data)

with col2:
    st.subheader("Répartition des transactions")
    pie_data = pd.DataFrame({
        'Type': ['Légitimes ✅', 'Fraudes 🚨'],
        'Nombre': [legit, fraudes]
    }).set_index('Type')
    st.bar_chart(pie_data)

st.divider()

# ── TABLEAU DES TRANSACTIONS ──
st.subheader("📋 Détail des transactions")

# Filtre
filtre = st.selectbox(
    "Filtrer par niveau de risque :",
    ["Toutes", "🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"]
)

display_df = df.copy()
if filtre != "Toutes":
    display_df = display_df[display_df['risk_level'] == filtre]

# Colonnes à afficher
display_df = display_df[[
    'Amount', 'fraud_probability', 'risk_level', 'is_fraud_predicted'
]].rename(columns={
    'Amount': 'Montant ($)',
    'fraud_probability': 'Probabilité de fraude',
    'risk_level': 'Niveau de risque',
    'is_fraud_predicted': 'Fraude détectée'
})

display_df['Probabilité de fraude'] = (
    display_df['Probabilité de fraude'] * 100
).round(2).astype(str) + '%'

display_df['Fraude détectée'] = display_df['Fraude détectée'].apply(
    lambda x: "🚨 OUI" if x == 1 else "✅ NON"
)

st.dataframe(display_df, use_container_width=True)

st.divider()

# ── FOOTER ──
st.caption(
    "Données : échantillon du dataset public "
    "[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · "
    "Modèle : XGBoost · AUC-ROC : 0.976"
)
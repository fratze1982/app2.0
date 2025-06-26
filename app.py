import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(layout="wide")

# CSV einlesen
df = pd.read_csv("rezeptdaten.csv", sep=";", decimal=",", encoding="utf-8")
df.columns = df.columns.str.strip()

# Zielgrößen
targets = [
    "Glanz 20", "Glanz 60", "Glanz 85",
    "Viskosität lowshear", "Viskosität midshear", "Brookfield",
    "Kosten Gesamt kg"
]
targets = [t for t in targets if t in df.columns]
for col in targets:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Input/Output
X = df.drop(columns=targets)
y = df[targets]

# Feature-Typen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X).fillna(0)
mask = y.notna().all(axis=1)
X_encoded_clean = X_encoded.loc[mask]
y_clean = y.loc[mask]

# Modell trainieren
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded_clean, y_clean.values)

# ------------------------------------
# STREAMLIT UI
# ------------------------------------
st.title("🎨 KI-Vorhersage und Analyse für Lackrezepturen")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    st.subheader("🔧 Rezeptur-Eingabe")
    for col in numerisch:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        if min_val < max_val:
            user_input[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            user_input[col] = st.number_input(col, value=mean_val)

    for col in kategorisch:
        options = sorted(df[col].dropna().unique())
        user_input[col] = st.selectbox(col, options)

# Eingabe vorbereiten
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# Vorhersage
prediction = modell.predict(input_encoded)[0]

with col2:
    st.subheader("🔮 Vorhergesagte Eigenschaften")
    for i, ziel in enumerate(targets):
        st.metric(label=ziel, value=round(prediction[i], 2))

# ------------------------------------
# Tabs für Analyse
# ------------------------------------
tab1, tab2 = st.tabs(["🔍 Einzelanalyse", "📈 Mehrfachvergleich"])

# ------------------------------------
# TAB 1 – Einzelanalyse
# ------------------------------------
with tab1:
    st.subheader("Einfluss eines Rohstoffs auf eine Zielgröße")

    feature_name = st.selectbox("🧪 Wähle Rohstoff", numerisch)
    target_name = st.selectbox("🎯 Wähle Zielgröße", targets)

    feature_index = list(X_encoded.columns).index(feature_name)
    target_index = targets.index(target_name)

    werte = np.linspace(df[feature_name].min(), df[feature_name].max(), 50)
    base_input = input_encoded.iloc[0].copy()
    sensi_inputs = pd.DataFrame([base_input] * len(werte))
    sensi_inputs[feature_name] = werte
    sensi_inputs = sensi_inputs[X_encoded.columns]
    sensi_preds = modell.predict(sensi_inputs)[:, target_index]

    fig, ax = plt.subplots(figsize=(8, 5))

    PartialDependenceDisplay.from_estimator(
        modell,
        X_encoded_clean,
        features=[feature_index],
        feature_names=X_encoded_clean.columns,
        target=target_index,
        ax=ax,
        line_kw={"label": "⛅ PDP (Global)", "color": "gray", "linestyle": "--"}
    )

    ax.plot(werte, sensi_preds, label="🔍 Sensitivität (Lokal)", color="blue")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(f"Einfluss von {feature_name} auf {target_name}")
    ax.legend()
    st.pyplot(fig)

# ------------------------------------
# TAB 2 – Mehrfachvergleich
# ------------------------------------
with tab2:
    st.subheader("Mehrere Rohstoffe vs. eine Zielgröße")

    selected_features = st.multiselect("🧪 Wähle mehrere Rohstoffe", numerisch, default=numerisch[:2])
    multi_target = st.selectbox("🎯 Zielgröße", targets)

    multi_target_index = targets.index(multi_target)

    fig, ax = plt.subplots(figsize=(10, 6))

    for feat in selected_features:
        feat_idx = list(X_encoded.columns).index(feat)

        try:
            PartialDependenceDisplay.from_estimator(
                modell,
                X_encoded_clean,
                features=[feat_idx],
                feature_names=X_encoded_clean.columns,
                target=multi_target_index,
                ax=ax,
                line_kw={"label": f"PDP – {feat}", "linestyle": "--", "alpha": 0.6}
            )
        except Exception as e:
            st.warning(f"PDP für {feat} konnte nicht berechnet werden: {e}")

        werte = np.linspace(df[feat].min(), df[feat].max(), 50)
        base_input = input_encoded.iloc[0].copy()
        sensi_inputs = pd.DataFrame([base_input] * len(werte))
        sensi_inputs[feat] = werte
        sensi_inputs = sensi_inputs[X_encoded.columns]
        sensi_preds = modell.predict(sensi_inputs)[:, multi_target_index]

        ax.plot(werte, sensi_preds, label=f"Sensitivität – {feat}")

    ax.set_ylabel(multi_target)
    ax.set_title(f"Einfluss mehrerer Rohstoffe auf {multi_target}")
    ax.legend()
    st.pyplot(fig)

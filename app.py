import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(layout="wide")

# CSV-Daten laden
df = pd.read_csv("rezeptdaten.csv", sep=";", decimal=",", encoding="utf-8")
df.columns = df.columns.str.strip()

# Zielgrößen definieren
targets = [
    "Glanz 20", "Glanz 60", "Glanz 85",
    "Viskosität lowshear", "Viskosität midshear", "Brookfield",
    "Kosten Gesamt kg"
]
targets = [t for t in targets if t in df.columns]

# Zielspalten in float konvertieren
for col in targets:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Eingabe- und Ausgabedaten trennen
X = df.drop(columns=targets)
y = df[targets]

# Feature-Typen erkennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.fillna(0)

# Nur vollständige Zeilen verwenden
mask = y.notna().all(axis=1)
X_encoded_clean = X_encoded.loc[mask]
y_clean = y.loc[mask]

# Modelltraining
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded_clean, y_clean.values)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎨 KI-Analyse für Lackrezepturen")

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

# ------------------------------
# Einflussanalyse
# ------------------------------
st.markdown("---")
st.header("📊 Einflussanalyse: Global vs. Dein Rezept")

# Feature und Zielgröße auswählen
feature_name = st.selectbox("🔍 Rohstoff wählen", numerisch)
target_name = st.selectbox("🎯 Zielgröße wählen", targets)

# Indexe berechnen
feature_index = list(X_encoded.columns).index(feature_name)
target_index = targets.index(target_name)

# Wertebereich für Sensitivitätsanalyse
werte = np.linspace(df[feature_name].min(), df[feature_name].max(), 50)
base_input = input_encoded.iloc[0].copy()
sensi_inputs = pd.DataFrame([base_input] * len(werte))
sensi_inputs[feature_name] = werte
sensi_inputs = sensi_inputs[X_encoded.columns]
sensi_preds = modell.predict(sensi_inputs)[:, target_index]

# Kombinierter Plot
fig, ax = plt.subplots(figsize=(8, 5))

# PDP global (durchschnittlich)
PartialDependenceDisplay.from_estimator(
    modell,
    X_encoded_clean,
    features=[feature_index],
    feature_names=X_encoded_clean.columns,
    target=target_index,
    ax=ax,
    line_kw={"label": "⛅ PDP (Globaler Mittelwert)", "color": "gray", "linestyle": "--"}
)

# Sensitivität (dein Rezept)
ax.plot(werte, sensi_preds, label="🔍 Sensitivität (Dein Rezept)", color="blue")
ax.set_xlabel(feature_name)
ax.set_ylabel(target_name)
ax.set_title(f"Einfluss von {feature_name} auf {target_name}")
ax.legend()

# Anzeige
st.pyplot(fig)

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
 
st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")
 
@st.cache_resource
def load_model():
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(iris.data, iris.target)
    return clf, iris.target_names
 
model, target_names = load_model()
 
# Sidebar sliders — Streamlit preserves their values automatically via widget keys
st.sidebar.header("🌸 Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.01, key="sl")
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.5, 0.01, key="sw")
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.01, key="pl")
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 0.2, 0.01, key="pw")
 
st.title("🌸 Iris Flower Classification App")
st.write("Predicts the species of an Iris flower based on flower measurements.")
st.success("✅ Model loaded successfully")
 
# ── Always predict live from current slider values (no button needed) ─────────
features  = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred_idx  = model.predict(features)[0]
pred_prob = model.predict_proba(features)[0]
species   = f"Iris-{target_names[pred_idx]}"
confidence = pred_prob[pred_idx] * 100
 
COLORS = {"setosa": "🔵", "versicolor": "🟢", "virginica": "🔴"}
icon = COLORS.get(target_names[pred_idx], "🌸")
 
st.subheader("Prediction")
st.info(f"{icon} **{species}** — {confidence:.1f}% confidence")
 
st.subheader("All Species Probabilities")
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]
for i, name in enumerate(target_names):
    pct = pred_prob[i] * 100
    cols[i].metric(f"Iris-{name}", f"{pct:.1f}%")
    cols[i].progress(float(pred_prob[i]))
 
st.subheader("Your Input")
st.table({
    "Measurement": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Value (cm)":  [sepal_length,   sepal_width,   petal_length,   petal_width],
})
 
st.divider()
st.caption("Built using Streamlit & Scikit-learn · Prediction updates live as you move sliders")
 

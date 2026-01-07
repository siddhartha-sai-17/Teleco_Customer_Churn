import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ---------------- Page Config ----------------
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# ---------------- Load CSS ----------------
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- Title ----------------
st.markdown("""
<div class="card">
<h1>🤖 Telco Customer Churn Prediction Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Telco_customer.csv")

df = load_data()

# ---------------- Encoding ----------------
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ---------------- Features & Target ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------- Train Test Split ----------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------------- Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ---------------- Prediction ----------------
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

# ---------------- Metrics ----------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# ========================= DASHBOARD LAYOUT =========================

left, center, right = st.columns([1.2, 2.5, 1.2])

# ---------------- LEFT: Dataset Preview ----------------
with left:
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

# ---------------- CENTER: NEW CUSTOMER PREDICTION ----------------
with center:
    st.subheader("🔮 Predict Churn for New Customer")

    tenure = st.slider(
        "Tenure (months)",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        int(df["tenure"].mean())
    )

    monthly = st.slider(
        "Monthly Charges",
        float(df["MonthlyCharges"].min()),
        float(df["MonthlyCharges"].max()),
        float(df["MonthlyCharges"].mean())
    )

    total = st.slider(
        "Total Charges",
        float(df["TotalCharges"].min()),
        float(df["TotalCharges"].max()),
        float(df["TotalCharges"].mean())
    )

    # Create input using mean of all features
    input_data = X.mean().to_dict()

    # Override 3 important features
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    input_df = pd.DataFrame([input_data])

    # Scale input
    input_scaled = scaler.transform(input_df)

    if st.button("🔮 Predict Churn"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"❌ Likely to Churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Likely to Stay (Probability: {1 - prob:.2f})")

# ---------------- RIGHT: Model Statistics ----------------
with right:
    st.subheader("📊 Model Statistics")
    st.metric("Accuracy", f"{accuracy:.2f}")
    st.metric("Precision", f"{report['1']['precision']:.2f}")
    st.metric("Recall", f"{report['1']['recall']:.2f}")
    st.metric("F1-Score", f"{report['1']['f1-score']:.2f}")

# ========================= GRAPHS SECTION =========================

st.markdown("---")

col1, col2 = st.columns(2)

# -------- Confusion Matrix --------
with col1:
    st.subheader("🧮 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        cmap="Blues",
        values_format="d",
        display_labels=["No", "Yes"],
        ax=ax_cm
    )
    st.pyplot(fig_cm)

# -------- ROC Curve --------
with col2:
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], '--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

# -------- Feature Importance --------
st.subheader("🧠 Feature Importance")

importance = model.coef_[0]
features = X.columns

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
ax_imp.barh(imp_df["Feature"], imp_df["Importance"])
ax_imp.invert_yaxis()
st.pyplot(fig_imp)

# ---------------- Footer ----------------
st.markdown("""
<hr>
<center>🚀 Telco Customer Churn Prediction System using Logistic Regression</center>
""", unsafe_allow_html=True)

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])


import streamlit as st
import pandas as pd
import joblib
import io
from data_checks_and_transformation import check_and_transform_data

# Load pretrained model
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")

model = load_model()

st.title("📊 Loan Default Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ✅ Step 1: Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # ✅ Step 2: Transform data
    try:
        df_transformed = check_and_transform_data(df)
        st.success("✅ Data checked & transformed successfully")
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    # ✅ Step 3: Make predictions
    preds = model.predict(df_transformed)

    # align predictions with transformed data
    df_result = df.loc[df_transformed.index].copy()
    df_result["prediction"] = preds

    st.write("### Dataset with Predictions")
    st.dataframe(df_result.head())

    # ✅ Step 4: If target exists, show metrics
    if "Default_status" in df:
        df_result['target']=df.loc[df_transformed.index].copy()['Default_status']
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        acc = accuracy_score(df_result["target"], preds)
        st.metric("Accuracy", f"{acc:.2%}")

        st.write("Confusion Matrix:")
        cm = confusion_matrix(df_result["target"], preds)
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        st.write("Classification Report:")
        report = classification_report(df_result["target"], preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.info("⚠️ No 'target' column found. Predictions generated without accuracy metrics.")

    # ✅ Step 5: Allow download
    st.write("### Download Predictions")

    # CSV
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_buffer.getvalue(),
        file_name="predictions.csv",
        mime="text/csv",
    )

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_result.to_excel(writer, index=False, sheet_name="Predictions")
    st.download_button(
        label="Download as Excel",
        data=excel_buffer.getvalue(),
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


# ------------- Load full pipeline ------------- #
@st.cache_resource
def load_pipeline(models_dir: Path = Path("model")):
    """
    Loads the saved full pipeline and optional metadata.
    Expects:
        model/full_pipeline.joblib
        model/metadata.json   (optional)
    """
    pipe_path = models_dir / "full_pipeline.joblib"
    meta_path = models_dir / "metadata.json"

    if not pipe_path.exists():
        raise FileNotFoundError("full_pipeline.joblib not found in 'model/' folder")

    pipe = joblib.load(pipe_path)

    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as fh:
            meta = json.load(fh)

    return pipe, meta


def predict_single(df: pd.DataFrame, pipe, meta):
    """
    Single-row DF -> prediction + probability (if classifier supports it)
    """
    if meta and "feature_names" in meta:
        feature_names = meta["feature_names"]
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required feature columns: {missing}")
        X = df[feature_names]
    else:
        X = df

    preds = pipe.predict(X)
    pred = preds[0]

    prob = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        # assume positive class is 'Y'
        if hasattr(pipe, "classes_") and "Y" in pipe.classes_:
            pos_idx = list(pipe.classes_).index("Y")
            prob = proba[pos_idx]
        else:
            prob = max(proba)

    return pred, prob


# ------------- Streamlit UI ------------- #

def main():
    st.set_page_config(
        page_title="Loan Approval Prediction App",
        page_icon="💰",
        layout="centered"
    )

    st.title("💰 Loan Approval Prediction App")
    st.write(
        "Fill the details below and the model will predict whether the loan is "
        "likely to be **Approved (Y)** or **Not Approved (N)**."
    )

    # load pipeline
    try:
        pipe, meta = load_pipeline(Path("model"))
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    with st.form("loan_form"):
        c1, c2 = st.columns(2)

        with c1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Married = st.selectbox("Married", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])

        with c2:
            ApplicantIncome = st.number_input(
                "Applicant Income", min_value=0, step=100, value=5000
            )
            CoapplicantIncome = st.number_input(
                "Coapplicant Income", min_value=0, step=100, value=0
            )
            LoanAmount = st.number_input(
                "Loan Amount (in thousands)", min_value=0, step=1, value=100
            )
            Loan_Amount_Term = st.number_input(
                "Loan Amount Term (in days)", min_value=0, step=12, value=360
            )
            Credit_History = st.selectbox("Credit History", [1.0, 0.0])
            Property_Area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"]
            )

        submitted = st.form_submit_button("Predict Loan Approval ✅")

    if submitted:
        # row exactly like training features
        row = {
            "Gender": Gender,
            "Married": Married,
            "Dependents": Dependents,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": Property_Area,
        }

        input_df = pd.DataFrame([row])

        try:
            pred, prob = predict_single(input_df, pipe, meta)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        if pred == "Y":
            st.success("✅ The model predicts: **Loan will be APPROVED (Y)**")
        else:
            st.error("❌ The model predicts: **Loan will NOT be APPROVED (N)**")

        if prob is not None:
            st.write(f"**Confidence (Approval Probability):** `{prob:.2%}`")

        with st.expander("See input data used for prediction"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()
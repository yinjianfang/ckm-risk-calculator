import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shap
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ===============================
# Page configuration
# ===============================

st.set_page_config(
    page_title="Advanced CKM Risk Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===============================
# Custom style
# ===============================

st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .top-banner {
        background: #263c50;
        color: white;
        padding: 34px 28px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 24px;
    }

    .top-banner h1 {
        font-size: 38px;
        line-height: 1.2;
        margin: 0;
        font-weight: 650;
    }

    .result-card {
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 26px 24px;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .prob-box {
        background: #d9f8e7;
        border-radius: 8px;
        padding: 18px;
        border-left: 5px solid #55c58a;
        font-size: 20px;
        text-align: center;
        font-weight: 700;
        margin: 16px 0 20px 0;
    }

    .risk-high {
        color: #c7254e;
        font-weight: 800;
    }

    .risk-low {
        color: #16805b;
        font-weight: 800;
    }

    .note-box {
        background: #f7f8fb;
        border-radius: 8px;
        padding: 16px 20px;
        color: #4b5563;
        font-size: 15px;
        margin-top: 14px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 13px;
    }

    section[data-testid="stSidebar"] {
        background-color: #eef2f4;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ===============================
# Model loading
# ===============================

@st.cache_resource
def load_model(model_path: str = "xgb_model.pkl"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


try:

    model = load_model()

    # SHAP explainer for probability-based individual interpretation
    explainer = shap.TreeExplainer(
        model,
        model_output="probability"
    )


except Exception as e:

    st.error(
        f"Failed to load model: {e}"
    )

    st.stop()



# ===============================
# Feature information
# ===============================

if hasattr(model, "feature_names_in_"):

    FEATURE_ORDER = list(model.feature_names_in_)

else:

    FEATURE_ORDER = [
        "DR",
        "lipid",
        "insulin",
        "age",
        "BUN",
        "UACR",
        "eGFR",
        "TG",
        "NEU",
        "Hb"
    ]



DEFAULTS = {

    "DR": 1,
    "lipid": 1,
    "insulin": 1,
    "age": 60.0,
    "eGFR": 71.02,
    "BUN": 8.005,
    "UACR": 225.405,
    "TG": 1.64,
    "NEU": 3.96,
    "Hb": 127.0,

}


LABELS = {

    "DR":
    "Diabetic Retinopathy",

    "lipid":
    "Lipid-lowering Medication",

    "insulin":
    "Insulin Therapy",

    "age":
    "Age (years)",

    "eGFR":
    "eGFR (mL/min/1.73 m²)",

    "BUN":
    "BUN (mmol/L)",

    "UACR":
    "UACR (mg/g)",

    "TG":
    "TG (mmol/L)",

    "NEU":
    "NEU (10⁹/L)",

    "Hb":
    "Hb (g/L)",

}
# ===============================
# Header
# ===============================

st.markdown(
    """
    <div class="top-banner">
        <h1>
        Advanced CKM Risk Identification<br>
        in Patients with Diabetic Kidney Disease
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)



# ===============================
# Sidebar inputs
# ===============================

st.sidebar.markdown(
    "## Patient Information"
)

st.sidebar.caption(
    "Enter patient characteristics and laboratory values."
)



# Binary variables

DR = (
    1
    if st.sidebar.selectbox(
        "Diabetic Retinopathy",
        ["No", "Yes"],
        index=1
    ) == "Yes"
    else 0
)


lipid = (
    1
    if st.sidebar.selectbox(
        "Lipid-lowering Medication",
        ["No", "Yes"],
        index=1
    ) == "Yes"
    else 0
)


insulin = (
    1
    if st.sidebar.selectbox(
        "Insulin Therapy",
        ["No", "Yes"],
        index=1
    ) == "Yes"
    else 0
)



st.sidebar.markdown("---")



age = st.sidebar.number_input(
    "Age (years)",
    min_value=18.0,
    max_value=110.0,
    value=float(DEFAULTS["age"]),
    step=1.0
)


eGFR = st.sidebar.number_input(
    "eGFR (mL/min/1.73 m²)",
    min_value=0.0,
    max_value=300.0,
    value=float(DEFAULTS["eGFR"]),
    step=1.0
)


BUN = st.sidebar.number_input(
    "BUN (mmol/L)",
    min_value=0.0,
    max_value=80.0,
    value=float(DEFAULTS["BUN"]),
    step=0.1
)


UACR = st.sidebar.number_input(
    "UACR (mg/g)",
    min_value=0.0,
    max_value=20000.0,
    value=float(DEFAULTS["UACR"]),
    step=10.0
)


TG = st.sidebar.number_input(
    "TG (mmol/L)",
    min_value=0.0,
    max_value=50.0,
    value=float(DEFAULTS["TG"]),
    step=0.1
)


NEU = st.sidebar.number_input(
    "NEU (10⁹/L)",
    min_value=0.0,
    max_value=50.0,
    value=float(DEFAULTS["NEU"]),
    step=0.1
)


Hb = st.sidebar.number_input(
    "Hb (g/L)",
    min_value=20.0,
    max_value=250.0,
    value=float(DEFAULTS["Hb"]),
    step=1.0
)



st.sidebar.markdown("---")



threshold = st.sidebar.slider(
    "High-risk threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help=
    "Patients with predicted probability greater than or equal to this threshold are classified as high risk."
)



calculate = st.sidebar.button(
    "Calculate Risk",
    type="primary",
    use_container_width=True
)



# ===============================
# Prepare input data
# ===============================


input_dict = {

    "DR": DR,
    "lipid": lipid,
    "insulin": insulin,
    "age": age,
    "eGFR": eGFR,
    "BUN": BUN,
    "UACR": UACR,
    "TG": TG,
    "NEU": NEU,
    "Hb": Hb,

}



# Fill missing values

for k, v in DEFAULTS.items():

    if (
        k not in input_dict
        or input_dict[k] is None
        or (
            isinstance(input_dict[k], float)
            and np.isnan(input_dict[k])
        )
    ):

        input_dict[k] = v



X = pd.DataFrame(
    [
        {
            feature: input_dict[feature]
            for feature in FEATURE_ORDER
        }
    ]
)



# ===============================
# Prediction functions
# ===============================


def predict_probability(model, data):

    """
    Return probability of advanced CKM.
    """

    proba = model.predict_proba(data)


    if hasattr(model, "classes_"):

        if 1 in list(model.classes_):

            idx = list(model.classes_).index(1)

        else:

            idx = 1

    else:

        idx = 1


    return float(
        proba[0, idx]
    )



# ===============================
# SHAP explanation function
# ===============================


def generate_shap_plot(
        model_explainer,
        patient_data
):


    shap_values = (
        model_explainer
        .shap_values(patient_data)
    )


    # Binary classification compatibility

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[1]



    if isinstance(
        model_explainer.expected_value,
        list
    ):

        base_value = (
            model_explainer.expected_value[1]
        )

    else:

        base_value = (
            model_explainer.expected_value
        )



    explanation = shap.Explanation(

        values=shap_values[0],

        base_values=base_value,

        data=patient_data.iloc[0],

        feature_names=patient_data.columns

    )



    fig = plt.figure(
        figsize=(8,5)
    )



    shap.plots.waterfall(

        explanation,

        show=False

    )



    return fig
    # ===============================
# Gauge figure
# ===============================


def make_gauge(
        probability,
        threshold_value
):

    score = probability * 100

    threshold_score = threshold_value * 100


    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=score,

            number={
                "suffix": "%",
                "font": {"size": 42}
            },

            title={
                "text": "Risk Gauge",
                "font": {"size": 28}
            },


            gauge={

                "axis": {
                    "range": [0,100],
                    "tickwidth":1,
                    "tickcolor":"#6b7280"
                },


                "bar": {

                    "color":
                    "#ef476f"
                    if probability >= threshold_value
                    else "#2a9d8f"

                },


                "bgcolor":"white",

                "borderwidth":0,


                "steps":[

                    {
                        "range":[0,threshold_score],
                        "color":"#d9f8e7"
                    },


                    {
                        "range":[threshold_score,100],
                        "color":"#ffe1ea"
                    }

                ],


                "threshold":{

                    "line":{
                        "color":"#111827",
                        "width":3
                    },

                    "thickness":0.75,

                    "value":threshold_score

                }

            }

        )

    )


    fig.update_layout(

        height=360,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=10
        )

    )


    return fig




# ===============================
# Main page
# ===============================


left, right = st.columns(
    [1.05,1.95],
    gap="large"
)



# ===============================
# Input summary
# ===============================


with left:


    st.markdown(
        "### Input Summary"
    )


    display_rows = []


    for feature in FEATURE_ORDER:


        val = input_dict[feature]


        if feature in [
            "DR",
            "lipid",
            "insulin"
        ]:

            val = (
                "Yes"
                if int(val)==1
                else "No"
            )


        display_rows.append(

            {
                "Variable":
                LABELS.get(
                    feature,
                    feature
                ),

                "Value":
                val
            }

        )


    st.dataframe(

        pd.DataFrame(display_rows),

        use_container_width=True,

        hide_index=True

    )



    st.markdown(

        """
        <p class='small-muted'>
        Binary variables are encoded as Yes=1 and No=0.
        Numeric default values are based on training-set medians.
        </p>
        """,

        unsafe_allow_html=True

    )





# ===============================
# Prediction result
# ===============================


with right:


    st.markdown(

        "<div class='result-card'>",

        unsafe_allow_html=True

    )


    st.markdown(
        "## Prediction Result"
    )



    if calculate:


        try:

            prob = predict_probability(
                model,
                X
            )


        except Exception as e:


            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()



        risk_label = (
            "High Risk"
            if prob >= threshold
            else "Low Risk"
        )


        risk_class = (
            "risk-high"
            if prob >= threshold
            else "risk-low"
        )



        st.markdown(

            f"""

            <div class='prob-box'>

            Predicted probability of advanced CKM:
            {prob*100:.1f}%


            <br>


            Classification:

            <span class='{risk_class}'>

            {risk_label}

            </span>

            </div>

            """,

            unsafe_allow_html=True

        )



        st.plotly_chart(

            make_gauge(
                prob,
                threshold
            ),

            use_container_width=True

        )



        # ===============================
        # Individual SHAP explanation
        # ===============================


        st.markdown("---")

        st.markdown(
            "### Individual Risk Explanation"
        )

        try:

            shap_fig = generate_shap_plot(
                explainer,
                X
            )

            st.pyplot(
                shap_fig,
                use_container_width=True
            )

            st.caption(
                "The SHAP waterfall plot illustrates how each feature contributes to this individual's predicted probability of advanced CKM risk."
            )

        except Exception as e:

            st.warning(
                f"SHAP explanation unavailable: {e}"
            )



        st.markdown(

            f"""

            <div class='note-box'>

            <b>Note:</b>

            With the current threshold of {threshold:.2f},

            a predicted probability ≥ {threshold:.2f}

            is classified as high risk.

            <br><br>

            This tool is intended for research demonstration

            and should not replace clinical judgment.

            </div>

            """,

            unsafe_allow_html=True

        )



    else:


        st.info(

            "Enter patient information on the left sidebar, then click **Calculate Risk**."

        )


        st.plotly_chart(

            make_gauge(
                0.0,
                threshold
            ),

            use_container_width=True

        )



    st.markdown(

        "</div>",

        unsafe_allow_html=True

    )





# ===============================
# Footer
# ===============================


st.markdown("---")


st.markdown(
    "### Model Variables"
)


st.markdown(

    """
    This XGBoost model uses 10 variables:

    diabetic retinopathy,
    lipid-lowering medication,
    insulin therapy,
    age,
    BUN,
    UACR,
    eGFR,
    TG,
    NEU,
    and Hb.
    """

)

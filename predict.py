import json
import joblib
import streamlit as st

with open("columns_attribute.json", "r") as f_in:
    col_attrs = json.load(f_in)
pipeline = joblib.load("pipeline.joblib")

st.title("Employee Attrition Prediction")
st.markdown(
    """This application predicts the probability of an employee leaving the company
    based on the provided input data. Fill in the details below to get the prediction."""
)

input_data = {}
for col_name, col_info in col_attrs.items():
    attr_type = col_info[0]

    if attr_type == "category":
        options = col_info[1]
        input_data[col_name] = st.selectbox(
            f"{col_name}", options
        )

    elif attr_type == "numeric":
        num_type = col_info[1]
        if num_type == "int":
            input_data[col_name] = st.number_input(
                f"{col_name}", step=1, format="%d"
            )
        elif num_type == "float":
            input_data[col_name] = st.number_input(
                f"{col_name}", step=0.1, format="%.1f"
            )

if st.button("Predict"):
    try:
        pred_proba = pipeline.predict_proba([input_data])[0, 1]
        st.markdown(
            f"<h2 style='text-align: center; color: blue;'>Estimated Attrition Probability for This Employee: {pred_proba * 100:.1f}%</h2>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {str(e)}")
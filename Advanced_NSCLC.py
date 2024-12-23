import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from scipy import stats

st.title("XGBoost-based Prediction of Immunochemotherapy Advanced Non-Small Cell Lung Carcinoma (NSCLC)")
# Create a function to generate HTML for person icons
def generate_person_icons(filled_count, total_count=100):
    # SVG person icon
    icon_svg = """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="7" r="4" stroke="black" stroke-width="2" fill="none"/>
      <path d="M4 21C4 16.6863 7.68629 13 12 13C16.3137 13 20 16.6863 20 21H4Z" stroke="black" stroke-width="2" fill="none"/>
    </svg>
    """
    
    # Replace fill attribute to change color
    filled_icon = icon_svg.replace('fill="none"', 'fill="lightblue"')
    empty_icon = icon_svg.replace('fill="none"', 'fill="gray"')

    # Generate the HTML for the icons
    icons_html = ''.join([filled_icon if i < filled_count else empty_icon for i in range(total_count)])
    return f"<div style='display: flex; flex-wrap: wrap; width: 480px;'>{icons_html}</div>"
# Load model
loaded_model = load('xgb_advanced_NSCLC_4%_model.joblib')

# Load saved Scaler
scaler = joblib.load('xgb_advanced_NSCLC_4%_scaler.joblib')

# Load validation set predictions
validation_predictions = np.load('D:\py-data\xgb_SCLC_5%_predictions.npy')
# Ensure validation_predictions is a 1D array
if validation_predictions.ndim > 1:
    validation_predictions = validation_predictions.ravel()

# Define feature order
features = ['D_D', 'ALP', 'CREA', 'MON_perc', 'BNP', 'INR']



with st.sidebar:
    st.header("Your information")
    alp = st.number_input('Alkaline Phosphatase (ALP, U/L)', min_value=0.0, max_value=5000.0, step=1.0, key='alp')
    inr = st.number_input('International Normalized Ratio (INR)', min_value=0.0, max_value=50.0, step=0.1, key='inr')
    d_d = st.number_input('D-dimer (D-D, mg/L)', min_value=0.0, max_value=50.0, step=0.1, key='d_d')
    crea = st.number_input('Creatinine (CREA, umol/L)', min_value=0.0, max_value=5000.0, step=1.0, key='crea')
    mon_perc = st.number_input('Percentage of monocytes (MON%)', min_value=0.0, max_value=100.0, step=1.0, key='mon_perc')
    bnp = st.number_input('B-type Natriuretic Peptide (BNP, pg/mL)', min_value=0.0, max_value=5000.0, step=1.0, key='bnp')





# Middle column: buttons
with st.container():
    st.write("")  # Placeholder

    # Use custom CSS for button styles
    st.markdown(
        """
        <style>
        .clear-button {
            background-color: transparent;
            color: black;
            border: none;
            text-decoration: underline;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            display: inline-block;
        }
        .clear-button:hover {
            color: red;
        }
        .clear-button:active {
            color: darkred;
        }
        </style>
        """, unsafe_allow_html=True)

    # Use HTML button
    st.markdown(
        """
        <a href="?reset=true" class="clear-button">Clear</a>
        """, unsafe_allow_html=True)

# If the prediction button is clicked
if st.button('Prediction'):
        # Prepare input data
        user_input = pd.DataFrame([[d_d,alp,crea,mon_perc,bnp,inr]], columns=features)
        # Normalize continuous features
        # user_continuous_input_normalized =scaler.fit(user_input[features])
        user_continuous_input_normalized = scaler.transform(user_input[features])
        
        # Combine normalized data back into the full input
        user_input_normalized = user_continuous_input_normalized.copy()

        # Get prediction probability
        prediction_proba = loaded_model.predict_proba(user_input)[:, 1][0]
        st.write(prediction_proba)
        prediction_percentage = round(prediction_proba * 100)

        # Combine user prediction with validation predictions
        combined_predictions = np.concatenate([validation_predictions, np.array([prediction_proba])])

        # Calculate standard deviation and confidence interval
        std_dev = np.std(combined_predictions)
        confidence_level = 0.95
        degrees_of_freedom = len(combined_predictions) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_critical * (std_dev / np.sqrt(len(combined_predictions)))
        lower_bound_percentage = max(prediction_percentage - margin_of_error * 100, 0)
        upper_bound_percentage = min(prediction_percentage + margin_of_error * 100, 100)

        lower_bound_percentage = round(lower_bound_percentage)
        upper_bound_percentage = round(upper_bound_percentage)
    
        # Right column: show prediction results
        with st.container():
            st.header("Your result")
            st.markdown(f"The probability that advanced NSCLC patients benefit from immunochemotherapy is (95% confidence interval):")
            result_html = f"""
            <div style="display: flex; align-items: center;">
                <span style="color:red; font-weight:bold; font-size:48px;">{prediction_percentage}%</span>
                <span style="margin-left: 10px;">({lower_bound_percentage}% to {upper_bound_percentage}%)</span>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
            # Use the function to generate icons based on prediction
            icons_html = generate_person_icons(prediction_percentage)

            # Display the generated icons
            st.markdown(f"""
                <div style="display: flex; align-items: center;">
                </div>
                <div>
                    {icons_html}
                </div>
            """, unsafe_allow_html=True)
            
            # Show additional information
            st.write(f"This result predicts how likely you are to benefit from immunochemotherapy. The probability means that out of 100 patients with similar characteristics, approximately {prediction_percentage}% may benefit from this therapy. More specifically, we're 95% confident that {lower_bound_percentage} to {upper_bound_percentage} out of 100 patients may benefit from this therapy, based on our training data. However, it's important to recognize that this is just a rough ballpark estimate. Individual patient outcomes can vary significantly, and a healthcare provider can provide a more precise assessment, taking into account a broader range of factors and personal medical history.")
            st.markdown(f"<span style='color:red;'>Disclaimer:</span> This tool is provided for informational purposes only and should NOT be considered as medical advice or a substitute for professional consultation. Users should seek proper medical counsel and discuss their treatment options with a qualified healthcare provider.", unsafe_allow_html=True)





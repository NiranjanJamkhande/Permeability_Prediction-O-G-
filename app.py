import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
joblib_file = "xgboost_model.pkl"
xgb = joblib.load(joblib_file)

st.set_page_config(
    page_title='Permeability Prediction',
    page_icon='✅',
    layout='wide'
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        margin-top: 0;
        padding-top: 0;
    }
    .subheader {
        font-size: 28px;
        font-weight: bold;
        color: #555;
        margin-top: 20px;
        padding-top: 0;
    }
    .content {
        margin-top: 30px;
        padding-top: 0;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    .stDataFrame {
        font-size: 20px; /* Adjust font size if needed */
        color: #333;
    }
    .stDataFrame table th {
        font-weight: bold !important;
        font-size: 10px; /* Adjust font size if needed */
        color: black;
        white-space: nowrap; /* Prevent text wrapping */
        text-align: centre; /* Center-align the text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 style="text-align: center;" class="title">Permeability Prediction</h1>', unsafe_allow_html=True)

st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)  # Adjust the margin value as needed

# Sidebar for file upload
st.sidebar.header("Upload CSV")
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(st.session_state.uploaded_file)

    # Set the depth column as the index
    if 'Depth' in data.columns:
        data.set_index('Depth', inplace=True)
        data.index.name = 'Depth'  # Ensure the index has a name

    # Read the other dataframe with the actual column
    actual_data = pd.read_csv("Comparing_csv.csv")

    # Add the 'actual column' from the other dataframe to input_data
    data['Actual Permeability'] = pd.Series(actual_data['Actual Permeability'].values, index=data.index)

    # Assuming the CSV has columns:
    input_data = data[['Acoustic (AC)', 'Density Log (DEN)', 'Gamma Ray (GR)', 'Neutron (NEU)', 'Photoelectric Absorption Factor (PEF)', 'Density Correction (DENC)', 'Deep Resistivity (RDEP)', 'Porosity', 'Grain Density']]

    # Make predictions
    predictions = xgb.predict(input_data)

    # Round predictions to 4 decimal places
    predictions = [round(p, 4) for p in predictions]

    # Add predictions to the dataframe
    data['Predicted Permeability'] = pd.Series(predictions, index=data.index)

    # Round the entire dataframe to 4 decimal places
    numeric_columns = data.select_dtypes(include='number').columns
    data[numeric_columns] = data[numeric_columns].round(4)

    # Ensure DataFrame is formatted correctly for display
    def format_dataframe(df):
        return df.style.format({
            col: '{:.4f}' for col in df.select_dtypes(include='number').columns
        }).set_table_attributes('style="width: 100%;"')  # Ensure the table takes full width

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Actual Permeability'], mode='lines', name="Actual Permeability", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=data.index, y=data['Predicted Permeability'], mode='lines', name="Predicted Permeability", line=dict(color='red')))
    fig1.update_layout(
        title={
            'text': "Actual vs Predicted Permeability",
            'x': 0.5,  # Center the title
            'xanchor': 'center',  # Center the title horizontally
            'yanchor': 'top'
        },
        title_font=dict(family="Arial, sans-serif", size=16, color="black", weight="bold"),  # Make title bold
        xaxis_title="Depth (ft)",
        yaxis_title="Permeability",
        xaxis_title_font=dict(family="Arial, sans-serif", size=18, color="black", weight="bold"),  # Bold and dark x-axis title
        yaxis_title_font=dict(family="Arial, sans-serif", size=18, color="black", weight="bold"),  # Bold and dark y-axis title
        legend_title="Legend",
        template="plotly_white",
        width=600,  # Adjust width as needed
        height=400,  # Adjust height as needed
        margin=dict(l=0, r=0, t=40, b=60),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for the plot area
        paper_bgcolor='rgba(0,0,0,0)'   # Transparent background for the entire figure
    )

    # Layout for plot and table
    col1, col2 = st.columns([2, 2])  # Adjust columns width
    with col1:
        st.dataframe(format_dataframe(data), height=400)
    with col2:
        st.plotly_chart(fig1, use_container_width=True)

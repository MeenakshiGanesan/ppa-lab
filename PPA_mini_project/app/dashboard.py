import streamlit as st
import pandas as pd
import plotly.express as px
from src.evaluate import get_all_results

st.set_page_config(
    page_title="Experiment 8 - Demonstration of Predictive Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body {
    background-color: #f4f4f9;
    font-family: 'Arial', sans-serif;
}
.header-box {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-card {
    background-color: #e2e8f0;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: #2d3748;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-card h3 {
    margin: 0;
    font-size: 1.5rem;
}
.metric-card p {
    margin: 0;
    font-size: 1rem;
    color: #4a5568;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>Experiment 8 - Demonstration of Predictive Models</h1>
    <p>Comparison of Decision Tree, KNN, and Neural Network</p>
</div>
""", unsafe_allow_html=True)

def fetch_results():
    results = get_all_results()
    return pd.DataFrame(results).T

df = fetch_results()

st.sidebar.markdown("""
<div style='text-align: center;'>
    <h3 style='color: #2d3748;'>Select Dataset</h3>
</div>
""", unsafe_allow_html=True)

selected_dataset = st.sidebar.selectbox(
    "",  # Removed redundant label above the dropdown
    ["All"] + list(df.index)
)

st.markdown("---")

if selected_dataset == "All":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Comparison Table")
    styled_df = df.style.format("{:.2f}")
    st.dataframe(styled_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Performance Visualization")
    df_melted = df.reset_index().melt(
        id_vars="index",
        var_name="Model",
        value_name="Accuracy"
    ).rename(columns={"index": "Dataset"})
    fig = px.bar(
        df_melted,
        x="Dataset",
        y="Accuracy",
        color="Model",
        barmode="group",
        height=400,
        color_discrete_sequence=px.colors.sequential.Blues_r  
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"{selected_dataset} Analysis")

    row = df.loc[selected_dataset]

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
        <h3>Decision Tree</h3>
        <p>{:.2f}</p>
    </div>
    """.format(row['Decision Tree']), unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
        <h3>KNN</h3>
        <p>{:.2f}</p>
    </div>
    """.format(row['KNN']), unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
        <h3>Neural Network</h3>
        <p>{:.2f}</p>
    </div>
    """.format(row['Neural Network']), unsafe_allow_html=True)

    single_df = pd.DataFrame({
        "Model": row.index,
        "Accuracy": row.values
    })

    fig = px.bar(
        single_df,
        x="Model",
        y="Accuracy",
        color="Model",
        height=300,
        color_discrete_sequence=px.colors.sequential.Blues_r  
    )
    st.plotly_chart(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Best Model per Dataset")
best_models = df.idxmax(axis=1)
best_df = pd.DataFrame({
    "Dataset": best_models.index,
    "Best Model": best_models.values
})

st.dataframe(best_df, use_container_width=True)

st.subheader("Insights")
st.markdown("""
<div style='background-color: #ebf8ff; padding: 15px; border-radius: 10px;'>
    <ul>
        <li><b>Decision Trees</b> dominate structured datasets</li>
        <li><b>KNN</b> performs well for behavioral patterns</li>
        <li><b>Neural Networks</b> provide consistent performance</li>
        <li><b>Trade dataset</b> is complex → lower accuracy expected</li>
    </ul>
</div>
""", unsafe_allow_html=True)

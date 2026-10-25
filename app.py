from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TellCo User Analytics", layout="wide")
st.title("TellCo — Telecom User Analytics Dashboard")
st.caption("Customer overview • engagement • network experience • satisfaction")
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"outputs"/"final_customer_scores.csv"
if not DATA.exists():
    st.warning("No analysis outputs found yet. Run the pipeline first: python -m telecom_analytics.pipeline --input data.xlsx --output outputs")
    st.stop()
df=pd.read_csv(DATA)
user_col="MSISDN/Number"
c1,c2,c3,c4=st.columns(4)
c1.metric("Customers",f"{df[user_col].nunique():,}" if user_col in df else "N/A")
c2.metric("Average sessions",f"{df['Sessions Frequency'].mean():,.1f}" if 'Sessions Frequency' in df else "N/A")
c3.metric("Average traffic (GB)",f"{df['Total Traffic (Bytes)'].mean()/1e9:,.2f}" if 'Total Traffic (Bytes)' in df else "N/A")
c4.metric("Avg satisfaction score",f"{df['Satisfaction Score'].mean():,.2f}" if 'Satisfaction Score' in df else "N/A")
search=st.text_input("Filter customer ID")
view=df.copy()
if search and user_col in view: view=view[view[user_col].astype(str).str.contains(search,case=False,na=False)]
st.subheader("Customer score table")
st.dataframe(view.head(1000),use_container_width=True)
if "Engagement Cluster" in df:
    st.subheader("Engagement cluster counts")
    st.bar_chart(df["Engagement Cluster"].value_counts().sort_index())
if "Satisfaction Score" in df:
    st.subheader("Satisfaction score distribution")
    st.bar_chart(df["Satisfaction Score"].dropna().reset_index(drop=True))
for file,title in [("top_10_handsets.csv","Top 10 handsets"),("top_3_manufacturers.csv","Top manufacturers"),("application_totals.csv","Application data volume")]:
    p=ROOT/"outputs"/file
    if p.exists():
        st.subheader(title); st.dataframe(pd.read_csv(p),use_container_width=True)
st.sidebar.info("Run the analysis pipeline before opening this dashboard. Results are descriptive and do not establish financial valuation on their own.")

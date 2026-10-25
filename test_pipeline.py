import pandas as pd
from telecom_analytics.pipeline import clean_data, add_derived_columns, aggregate_users

def test_clean_data_imputes_numeric_and_categorical_missing():
    df=pd.DataFrame({'n':[1.0,None,3.0], 'cat':['a',None,'a']})
    out=clean_data(df)
    assert out['n'].isna().sum()==0
    assert out['cat'].isna().sum()==0

def test_derived_app_and_traffic_columns():
    df=pd.DataFrame({'MSISDN/Number':[1], 'Dur. (ms)':[100], 'Total DL (Bytes)':[200], 'Total UL (Bytes)':[50], 'Social Media DL (Bytes)':[10], 'Social Media UL (Bytes)':[5]})
    out=add_derived_columns(df)
    assert out.loc[0,'Social Media data (Bytes)']==15
    assert out.loc[0,'Total Traffic (Bytes)']==250

def test_aggregate_users_counts_sessions():
    df=pd.DataFrame({'MSISDN/Number':[1,1,2], 'Dur. (ms)':[100,200,50], 'Total DL (Bytes)':[100,200,50], 'Total UL (Bytes)':[10,20,5]})
    out=aggregate_users(df)
    row=out[out['MSISDN/Number']==1].iloc[0]
    assert row['Sessions Frequency']==2
    assert row['Dur. (ms)']==300

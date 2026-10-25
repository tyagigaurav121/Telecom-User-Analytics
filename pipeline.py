"""End-to-end reusable telecom analytics pipeline for the supplied xDR workbook."""
from __future__ import annotations
import argparse, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

APP_COLS = {
    "Social Media": ("Social Media DL (Bytes)", "Social Media UL (Bytes)"),
    "Google": ("Google DL (Bytes)", "Google UL (Bytes)"),
    "Email": ("Email DL (Bytes)", "Email UL (Bytes)"),
    "YouTube": ("Youtube DL (Bytes)", "Youtube UL (Bytes)"),
    "Netflix": ("Netflix DL (Bytes)", "Netflix UL (Bytes)"),
    "Gaming": ("Gaming DL (Bytes)", "Gaming UL (Bytes)"),
    "Other": ("Other DL (Bytes)", "Other UL (Bytes)"),
}
USER_COL = "MSISDN/Number"
DURATION = "Dur. (ms)"
DL, UL = "Total DL (Bytes)", "Total UL (Bytes)"
TCP_COLS = ["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]
RTT_COLS = ["Avg RTT DL (ms)", "Avg RTT UL (ms)"]
TP_COLS = ["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the first sheet from Excel or a CSV file."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    return pd.read_excel(p, sheet_name=0, engine="openpyxl")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric columns and impute numeric missing/outliers with column mean.

    Outliers are defined using the 1.5*IQR rule and replaced by the column mean, as
    required by the project brief. Categorical missing values are filled with mode.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    id_like = {"Bearer Id", "IMSI", "MSISDN/Number", "IMEI"}
    for col in out.columns:
        if col in id_like:
            # IDs are identifiers, not measurements: never impute or outlier-replace them.
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean = s.mean()
            if pd.notna(mean):
                s = s.fillna(mean)
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
                    s.loc[mask] = mean
            out[col] = s
        else:
            mode = out[col].mode(dropna=True)
            if len(mode):
                out[col] = out[col].fillna(mode.iloc[0])
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for app, (d, u) in APP_COLS.items():
        if d in out and u in out:
            out[f"{app} data (Bytes)"] = out[d].fillna(0) + out[u].fillna(0)
    if DL in out and UL in out:
        out["Total Traffic (Bytes)"] = out[DL].fillna(0) + out[UL].fillna(0)
    if DURATION in out:
        out["Session Duration (ms)"] = out[DURATION]
    out["TCP Retransmission (Bytes)"] = out[[c for c in TCP_COLS if c in out]].mean(axis=1) if any(c in out for c in TCP_COLS) else np.nan
    out["RTT (ms)"] = out[[c for c in RTT_COLS if c in out]].mean(axis=1) if any(c in out for c in RTT_COLS) else np.nan
    out["Throughput (kbps)"] = out[[c for c in TP_COLS if c in out]].mean(axis=1) if any(c in out for c in TP_COLS) else np.nan
    return out


def aggregate_users(df: pd.DataFrame) -> pd.DataFrame:
    d = add_derived_columns(df)
    if USER_COL not in d:
        raise ValueError(f"Required customer identifier missing: {USER_COL}")
    d = d[d[USER_COL].notna()].copy()
    agg = {DURATION: "sum", "Total Traffic (Bytes)": "sum"}
    for c in [DL, UL] + [f"{a} data (Bytes)" for a in APP_COLS] + ["TCP Retransmission (Bytes)", "RTT (ms)", "Throughput (kbps)"]:
        if c in d.columns:
            agg[c] = "mean" if c in ["TCP Retransmission (Bytes)", "RTT (ms)", "Throughput (kbps)"] else "sum"
    users = d.groupby(USER_COL, dropna=True).agg(agg).reset_index()
    freq = d.groupby(USER_COL).size().rename("Sessions Frequency").reset_index()
    users = users.merge(freq, on=USER_COL, how="left")
    if "Handset Type" in d:
        handset = d.groupby(USER_COL)["Handset Type"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown").reset_index()
        users = users.merge(handset, on=USER_COL, how="left")
    return users


def _cluster(data: pd.DataFrame, cols: list[str], k: int, random_state: int = 42):
    x = data[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(xs)
    return labels, scaler, model, xs


def run_pipeline(input_path: str | Path, output_dir: str | Path) -> dict:
    start = time.time(); outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    raw = load_data(input_path)
    # Keep a raw-quality summary before imputation.
    quality = pd.DataFrame({"column": raw.columns, "dtype": [str(x) for x in raw.dtypes], "missing_count": raw.isna().sum().values, "missing_pct": (raw.isna().mean().values*100).round(3)})
    quality.to_csv(outdir / "data_quality.csv", index=False)
    cleaned = clean_data(raw)
    d = add_derived_columns(cleaned)
    users = aggregate_users(cleaned)
    # Overview tables
    if "Handset Type" in d:
        d["Handset Type"].value_counts().head(10).rename_axis("Handset Type").reset_index(name="Sessions").to_csv(outdir/"top_10_handsets.csv",index=False)
    if "Handset Manufacturer" in d:
        topm = d["Handset Manufacturer"].value_counts().head(3)
        topm.rename_axis("Manufacturer").reset_index(name="Sessions").to_csv(outdir/"top_3_manufacturers.csv",index=False)
        d[d["Handset Manufacturer"].isin(topm.index)].groupby(["Handset Manufacturer","Handset Type"]).size().reset_index(name="Sessions").sort_values(["Handset Manufacturer","Sessions"],ascending=[True,False]).groupby("Handset Manufacturer").head(5).to_csv(outdir/"top_5_handsets_per_manufacturer.csv",index=False)
    users.to_csv(outdir/"user_aggregates.csv", index=False)
    # Descriptive statistics and dispersion
    num = d.select_dtypes(include=np.number)
    num.describe().T.to_csv(outdir/"descriptive_statistics.csv")
    dispersion = pd.DataFrame({"mean":num.mean(),"median":num.median(),"std":num.std(),"variance":num.var(),"min":num.min(),"q1":num.quantile(.25),"q3":num.quantile(.75),"max":num.max(),"range":num.max()-num.min(),"iqr":num.quantile(.75)-num.quantile(.25)})
    dispersion.to_csv(outdir/"dispersion_metrics.csv")
    app_user_cols = [f"{a} data (Bytes)" for a in APP_COLS if f"{a} data (Bytes)" in users]
    if app_user_cols:
        users[app_user_cols].corr().to_csv(outdir/"application_correlation.csv")
        app_totals = users[app_user_cols].sum().sort_values(ascending=False)
        app_totals.rename_axis("Application").reset_index(name="Total Data (Bytes)").to_csv(outdir/"application_totals.csv",index=False)
        for app_col in app_user_cols:
            safe = app_col.lower().replace(" data (bytes)", "").replace(" ", "_")
            users.nlargest(10, app_col)[[USER_COL, app_col]].to_csv(outdir/f"top_10_users_{safe}.csv", index=False)
        # PCA of per-user application usage
        pca_cols = app_user_cols
        xp = users[pca_cols].fillna(0)
        pca_scaler = StandardScaler(); xp_s = pca_scaler.fit_transform(xp)
        pca = PCA(n_components=min(4, xp_s.shape[1], xp_s.shape[0]))
        pcs = pca.fit_transform(xp_s)
        pd.DataFrame(pcs,columns=[f"PC{i+1}" for i in range(pcs.shape[1])]).to_csv(outdir/"pca_scores.csv",index=False)
        pd.DataFrame({"component":[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],"explained_variance_ratio":pca.explained_variance_ratio_}).to_csv(outdir/"pca_variance.csv",index=False)
    # Duration deciles: longest-duration group at decile 10
    if DURATION in users and "Total Traffic (Bytes)" in users:
        users["Duration Decile"] = pd.qcut(users[DURATION].rank(method="first"), 10, labels=False) + 1
        users.groupby("Duration Decile").agg(Customers=(USER_COL,"count"),Total_Duration_ms=(DURATION,"sum"),Total_Traffic_Bytes=("Total Traffic (Bytes)","sum")).reset_index().to_csv(outdir/"duration_deciles.csv",index=False)
    # Engagement: frequency, duration and traffic
    eng_cols = [c for c in ["Sessions Frequency", DURATION, "Total Traffic (Bytes)"] if c in users]
    if len(eng_cols)==3 and len(users)>=3:
        labels, eng_scaler, eng_model, eng_x = _cluster(users, eng_cols, 3)
        users["Engagement Cluster"] = labels
        # low engagement cluster is lowest centroid average in standardized metric space
        low_cluster = int(np.argmin(eng_model.cluster_centers_.mean(axis=1)))
        users["Engagement Score"] = np.linalg.norm(eng_x-eng_model.cluster_centers_[low_cluster],axis=1)
        for col in eng_cols:
            users.nlargest(10,col)[[USER_COL,col]].to_csv(outdir/f"top_10_{col.lower().replace(' ','_').replace('(','').replace(')','')}.csv",index=False)
        users.groupby("Engagement Cluster")[eng_cols].agg(["min","max","mean","sum"]).to_csv(outdir/"engagement_cluster_stats.csv")
        # elbow values
        # Elbow fits on a deterministic sample for practical runtime on large customer bases.
        ks=[]
        elbow_x = eng_x if len(eng_x) <= 10000 else eng_x[np.linspace(0, len(eng_x)-1, 10000, dtype=int)]
        for k in range(1,min(10,len(elbow_x))):
            km=KMeans(n_clusters=k,random_state=42,n_init=5, max_iter=150).fit(elbow_x); ks.append({"k":k,"inertia":km.inertia_})
        pd.DataFrame(ks).to_csv(outdir/"engagement_elbow.csv",index=False)
    # Task 3.2: top, bottom and most frequent observed values for network metrics.
    for metric, cols in [("TCP", TCP_COLS), ("RTT", RTT_COLS), ("Throughput", TP_COLS)]:
        available = [c for c in cols if c in d.columns]
        if available:
            vals = pd.concat([d[c] for c in available], ignore_index=True).dropna()
            vals.value_counts().head(10).rename_axis("Value").reset_index(name="Frequency").to_csv(outdir/f"{metric.lower()}_most_frequent_values.csv", index=False)
            vals.nlargest(10).reset_index(drop=True).rename("Value").to_frame().to_csv(outdir/f"{metric.lower()}_top_values.csv", index=False)
            vals.nsmallest(10).reset_index(drop=True).rename("Value").to_frame().to_csv(outdir/f"{metric.lower()}_bottom_values.csv", index=False)
    # Experience clustering on aggregated user values
    exp_cols = [c for c in ["TCP Retransmission (Bytes)","RTT (ms)","Throughput (kbps)"] if c in users]
    if len(exp_cols)==3 and len(users)>=3:
        labels, exp_scaler, exp_model, exp_x = _cluster(users, exp_cols, 3)
        users["Experience Cluster"] = labels
        # worst cluster: highest normalized retransmission + RTT and lower throughput
        center=exp_model.cluster_centers_
        worst = int(np.argmax(center[:,0] + center[:,1] - center[:,2]))
        users["Experience Score"] = np.linalg.norm(exp_x-center[worst],axis=1)
        users.groupby("Experience Cluster")[exp_cols].mean().to_csv(outdir/"experience_cluster_means.csv")
        if "Handset Type" in users:
            users.groupby("Handset Type")["Throughput (kbps)"].mean().sort_values(ascending=False).head(30).to_csv(outdir/"throughput_by_handset.csv")
            users.groupby("Handset Type")["TCP Retransmission (Bytes)"].mean().sort_values(ascending=False).head(30).to_csv(outdir/"tcp_by_handset.csv")
    # Satisfaction: average of distance-based scores; regression estimates target for demonstration
    if "Engagement Score" in users and "Experience Score" in users:
        users["Satisfaction Score"]=(users["Engagement Score"]+users["Experience Score"])/2
        users.nlargest(10,"Satisfaction Score")[[USER_COL,"Engagement Score","Experience Score","Satisfaction Score"]].to_csv(outdir/"top_10_satisfied_customers.csv",index=False)
        sx=users[["Engagement Score","Experience Score"]].fillna(0)
        if len(users)>=5:
            # Train/test split is deterministic; target is defined from component scores, so this is a pipeline check not an independent outcome prediction.
            model=LinearRegression().fit(sx,users["Satisfaction Score"])
            pred=model.predict(sx)
            joblib.dump(model,outdir/"satisfaction_regression.joblib")
            metrics={"r2_in_sample":float(r2_score(users["Satisfaction Score"],pred)),"rmse_in_sample":float(mean_squared_error(users["Satisfaction Score"],pred)**0.5),"note":"In-sample fit against a score mathematically defined from the input scores; not independent validation."}
            (outdir/"model_metrics.json").write_text(json.dumps(metrics,indent=2))
        slabels, _, smodel, _ = _cluster(users,["Engagement Score","Experience Score"],2)
        users["Satisfaction Cluster"] = slabels
        users.groupby("Satisfaction Cluster")[["Satisfaction Score","Experience Score"]].mean().to_csv(outdir/"satisfaction_cluster_summary.csv")
    users.to_csv(outdir/"final_customer_scores.csv",index=False)
    # model/run tracking metadata
    metadata={"source_file":Path(input_path).name,"code_version":"1.0.0","python_version":platform.python_version(),"rows_input":int(len(raw)),"columns_input":int(raw.shape[1]),"customers_aggregated":int(users[USER_COL].nunique()),"start_epoch":start,"end_epoch":time.time(),"duration_seconds":round(time.time()-start,2),"artifacts":sorted(p.name for p in outdir.iterdir())}
    (outdir/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    return metadata

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description="Run telecom user analytics")
    ap.add_argument("--input",default="data.csv")
    ap.add_argument("--output",default="outputs")
    args=ap.parse_args()
    print(json.dumps(run_pipeline(args.input,args.output),indent=2))

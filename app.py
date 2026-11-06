"""Student Academic Performance Predictor (KNN vs Naive Bayes) - Streamlit app.

Run:  streamlit run app.py
Needs student_data.csv in the same folder (or upload a CSV in the sidebar).
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = Path(__file__).parent / "student_data.csv"
DEFAULT_TARGET = "G3"
DEFAULT_FEATURES = ["absences", "studytime", "failures", "G1", "G2"]
BANDS = ["Low", "Medium", "High"]


def knn(k):
    return Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(k))])


@st.cache_resource(show_spinner=False)
def run_pipeline(csv_text, features, target):
    """Same pipeline as student_performance_knn_nb.py (features/target selectable)."""
    features = list(features)
    raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    df = raw[features + [target]].drop_duplicates().dropna(subset=[target])
    df["performance"], bins = pd.qcut(df[target], 3, labels=BANDS,
                                      duplicates="drop", retbins=True)
    band_table = (df.groupby("performance", observed=True)[target]
                    .agg(["min", "max", "count"]).reindex(BANDS))
    band_table.columns = [f"Lowest {target}", f"Highest {target}", "Students"]

    X, y = df[features], df["performance"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    med, lo, hi = Xtr.median(), Xtr.quantile(.01), Xtr.quantile(.99)
    Xtr = Xtr.fillna(med).clip(lo, hi, axis=1)
    Xte = Xte.fillna(med).clip(lo, hi, axis=1)

    ks = list(range(1, 26))
    cv_k = [cross_val_score(knn(k), Xtr, ytr, cv=5).mean() for k in ks]
    best_k = ks[int(np.argmax(cv_k))]
    models = {f"KNN (k={best_k})": knn(best_k),
              "Naive Bayes": Pipeline([("sc", StandardScaler()), ("m", GaussianNB())])}

    cv_res = {n: cross_val_score(m, Xtr, ytr, cv=5).mean() for n, m in models.items()}
    best = max(cv_res, key=cv_res.get)
    test_res, reports, cms = {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        test_res[name] = accuracy_score(yte, pred)
        reports[name] = classification_report(yte, pred)
        cms[name] = confusion_matrix(yte, pred, labels=BANDS)

    mdl = models[best]
    dash = Xte.copy()
    dash["Predicted"] = mdl.predict(Xte)
    low_idx = list(mdl.classes_).index("Low")
    dash["Risk_Prob_%"] = (mdl.predict_proba(Xte)[:, low_idx] * 100).round(1)
    alerts = dash[dash["Predicted"] == "Low"].sort_values("Risk_Prob_%", ascending=False)
    pred_dist = dash["Predicted"].value_counts().reindex(BANDS).fillna(0)

    return dict(n_raw=len(raw), n_clean=len(df), n_train=len(Xtr), n_test=len(Xte),
                band_table=band_table, ks=ks, cv_k=cv_k, best_k=best_k, models=models,
                cv_res=cv_res, test_res=test_res, best=best, reports=reports, cms=cms,
                alerts=alerts, pred_dist=pred_dist, lo=lo, hi=hi, med=med,
                features=features, target=target, stats=df[features].describe())


def predict_student(res, model_name, values):
    row = pd.DataFrame([values])[res["features"]].clip(res["lo"], res["hi"], axis=1)
    mdl = res["models"][model_name]
    probs = {str(c): float(p) * 100 for c, p in zip(mdl.classes_, mdl.predict_proba(row)[0])}
    return str(mdl.predict(row)[0]), probs


def fig_accuracy(res):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    names, vals = list(res["test_res"]), list(res["test_res"].values())
    bars = ax.bar(names, vals, color=["#3b82f6", "#10b981"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Test accuracy"); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def fig_k(res):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot(res["ks"], res["cv_k"], marker="o")
    ax.axvline(res["best_k"], color="red", ls="--", lw=1)
    ax.set_title(f"KNN: k vs CV accuracy (best k = {res['best_k']})", fontsize=10)
    ax.set_xlabel("k")
    fig.tight_layout()
    return fig


def fig_pie(res):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    d = res["pred_dist"]
    ax.pie(d.values, labels=d.index, autopct="%1.0f%%")
    ax.set_title(f"Predicted bands ({res['best']})", fontsize=10)
    fig.tight_layout()
    return fig


def fig_cm(res, name):
    m = res["cms"][name]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(m, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(BANDS)
    ax.set_yticks(range(3)); ax.set_yticklabels(BANDS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
    st.title("🎓 Student Academic Performance Predictor")
    st.caption("KNN vs Naive Bayes  |  Early-warning tool for mentors  |  "
               "SHAIK FATHIMA BEE (23ME1A5460)  |  Guide: Mr. R. Sri Ramu")

    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload a student CSV (optional)", type=["csv"])
        st.write("Default file: student_data.csv (Kaggle Student Performance Dataset).")

    if up is not None:
        csv_text = up.getvalue().decode("utf-8", errors="replace")
    elif CSV_PATH.exists():
        csv_text = CSV_PATH.read_text(encoding="utf-8")
    else:
        st.error("student_data.csv not found. Put it in the same folder as app.py or upload a CSV.")
        st.stop()

    try:
        raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        st.error(f"Could not read the file as a CSV: {e}")
        st.stop()
    num_cols = raw.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        st.error("The file needs at least two numeric columns (inputs and a numeric target).")
        st.stop()

    with st.sidebar:
        st.header("Columns")
        t_default = num_cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in num_cols else len(num_cols) - 1
        target = st.selectbox("Target column (final result / marks)", num_cols, index=t_default)
        f_options = [c for c in num_cols if c != target]
        f_default = [c for c in DEFAULT_FEATURES if c in f_options] or f_options[:5]
        features = st.multiselect("Input columns", f_options, default=f_default)
        st.write("The target is split into Low / Medium / High (three equal groups).")

    if not features:
        st.warning("Select at least one input column in the sidebar.")
        st.stop()

    try:
        with st.spinner("Training models (KNN k-tuning + Naive Bayes)..."):
            res = run_pipeline(csv_text, tuple(features), target)
    except Exception as e:
        st.error(f"Could not process this data: {e}")
        st.stop()

    names = list(res["models"])
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🎯 Predict a student",
                                      "⚠️ At-risk list", "ℹ️ About"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records used", f"{res['n_clean']}")
        c2.metric("Best k (KNN)", f"{res['best_k']}")
        c3.metric("Selected model (by CV)", res["best"])
        c4.metric("At-risk students", f"{len(res['alerts'])} of {res['n_test']}")
        st.subheader("Model comparison")
        tbl = pd.DataFrame({"Model": names,
                            "CV accuracy": [f"{res['cv_res'][n]:.3f}" for n in names],
                            "Test accuracy": [f"{res['test_res'][n]:.3f}" for n in names]})
        st.table(tbl.set_index("Model"))
        st.write("Both models perform almost the same. The model is selected by "
                 "cross-validation accuracy, and the test set is used only once.")
        st.subheader("Charts")
        g1, g2, g3 = st.columns(3)
        with g1:
            show(fig_accuracy(res))
        with g2:
            show(fig_k(res))
        with g3:
            show(fig_pie(res))
        st.subheader("Confusion matrices (test set)")
        m1, m2 = st.columns(2)
        with m1:
            show(fig_cm(res, names[0]))
        with m2:
            show(fig_cm(res, names[1]))
        st.subheader("What do Low / Medium / High mean?")
        st.write(f"The target ({res['target']}) is split into three equal-frequency bands (quantile bins).")
        st.table(res["band_table"])
        with st.expander("Classification reports"):
            for n in names:
                st.write(n)
                st.code(res["reports"][n])

    with tab2:
        st.subheader("Enter a student's details")
        left, right = st.columns(2)
        values = {}
        for i, f in enumerate(res["features"]):
            col = left if i % 2 == 0 else right
            lo_v, hi_v = float(res["stats"].loc["min", f]), float(res["stats"].loc["max", f])
            default = float(res["med"][f])
            values[f] = col.number_input(f, min_value=lo_v, max_value=hi_v, value=default)
        choice = st.radio("Model", names, index=names.index(res["best"]), horizontal=True)
        if st.button("Predict performance band"):
            band, probs = predict_student(res, choice, values)
            if band == "Low":
                st.error("Predicted band: LOW. This student is at risk. Mentor attention is advised.")
            elif band == "Medium":
                st.warning("Predicted band: MEDIUM. Monitor this student.")
            else:
                st.success("Predicted band: HIGH. Likely to perform well.")
            st.write("Probability of each band (%):")
            st.bar_chart(pd.Series(probs).reindex(BANDS))

    with tab3:
        st.subheader("Students predicted as Low (test set), highest risk first")
        st.write(f"Model used: {res['best']}")
        min_risk = st.slider("Show students with risk % of at least", 0, 100, 0)
        al = res["alerts"]
        al = al[al["Risk_Prob_%"] >= min_risk]
        st.write(f"{len(al)} students shown")
        st.dataframe(al)
        st.download_button("Download mentor_alerts.csv", data=al.to_csv().encode("utf-8"),
                           file_name="mentor_alerts.csv", mime="text/csv")

    with tab4:
        st.subheader("About this project")
        st.write("Title: A K-Nearest Neighbours and Naive Bayes Comparison for Student Academic Performance")
        st.write(f"Dataset: Kaggle Student Performance Dataset ({res['n_raw']} records in the file, "
                 f"{res['n_clean']} after removing duplicates; {res['n_train']} for training, "
                 f"{res['n_test']} for testing).")
        st.write(f"Inputs: {', '.join(res['features'])}. Target: {res['target']} converted to Low / Medium / High.")
        st.write("Method: stratified 80:20 split, preprocessing fitted on training data only, "
                 "standard scaling, k tuned by 5-fold cross-validation, model chosen by cross-validation.")
        st.write("Limitations: small dataset; earlier grades (such as G1 and G2) are strongly related to "
                 "the final grade, which makes prediction easier; bands are relative to this group of students.")


if __name__ == "__main__":
    main()

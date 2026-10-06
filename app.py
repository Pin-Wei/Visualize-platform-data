#!/usr/bin/env python


from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


DOMAINS_MAP = {
    "工作記憶": "WorkMem", 
    "情節記憶": "EpisMem", 
    "語言理解": "LangComp", 
    "語言產出": "LangProd", 
    "動作": "Motor"
}
DOMAINS = list(DOMAINS_MAP.values())
DEFAULT_DATA_PATH = next((Path(__file__).parent / "data").glob("*.csv"), None)


def load_data(data_path=DEFAULT_DATA_PATH):
    df = pd.read_csv(data_path)
    df.rename(columns=DOMAINS_MAP, inplace=True)
    df[DOMAINS] = df[DOMAINS].astype(float).replace(-1, np.nan)
    df.drop(columns=["Date", "Name", "PAD", "Corrected PAD", "Avg"], inplace=True)
    df["Session"] = df["Session"].astype("category")

    return df


def plot_cross_sectional(df, y_axis, session="All", age_min=None, age_max=None, pad=2):
    if session != "All":
        df = df.query(f"Session == {session}")
    if age_min is not None:
        df = df.query(f"Age >= {age_min}")
    if age_max is not None:
        df = df.query(f"Age <= {age_max}")
    assert len(df) > 0, "No data passed the filter"

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, x="Age", y=y_axis, hue="SID", 
        kind="strip", size=7, alpha=.5, palette="Set2", 
        native_scale=True, legend=False, zorder=1, 
        height=5, aspect=1.5, 
    )
    sns.regplot(
        data=df, x="Age", y=y_axis, 
        line_kws={"color": "k", "lw": 2},
        scatter=False, truncate=False, order=2, 
        ax=g.ax
    )
    g.ax.tick_params(axis="both", labelsize=14)
    g.set_xlabels(fontsize=16)
    g.set_ylabels(fontsize=16)
    g.ax.set_xlim(min(df["Age"]) - pad, max(df["Age"]) + pad)
    g.ax.set_ylim(0, 100 + pad)

    return g


def plot_longitudinal(df, y_axis, sid="All", age_min=None, age_max=None, pad=5):
    if sid != "All":
        df = df.query(f"SID == '{sid}'")
    if age_min is not None:
        df = df.query(f"Age >= {age_min}")
    if age_max is not None:
        df = df.query(f"Age <= {age_max}")
    assert len(df) > 0, "No data passed the filter"

    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(7.5, 5))
    sns.pointplot(
        df, x="Session", y=y_axis, hue="SID",
        markersize=7, alpha=.5, legend=False, 
        ax=ax
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.set_ylim(0, 100 + pad)

    return ax


def plot_corr_heatmap(df, col_type, session="All", age_min=None, age_max=None):
    if session != "All":
        df = df.query(f"Session == {session}")
    if age_min is not None:
        df = df.query(f"Age >= {age_min}")
    if age_max is not None:
        df = df.query(f"Age <= {age_max}")
    assert len(df) > 0, "No data passed the filter"

    if col_type == "domain scores":
        fig_size = (5, 5)
        kwargs = {"annot": True}
    else: # "raw scores"
        fig_size = (20, 20)
        kwargs = {"annot": False}

    sns.set_theme(style="white")
    _, ax = plt.subplots(figsize=fig_size)
    cormat = df.corr()
    mask = np.triu(np.ones_like(cormat, dtype=bool), k=0)
    sns.heatmap(
        cormat, mask=mask, square=True, 
        vmin=-1, vmax=1, cmap="coolwarm", cbar_kws={"shrink": 0.6}, 
        ax=ax, **kwargs
    )
    ax.tick_params(axis="both", labelsize=16)
    if col_type == "domain scores":
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    else: # "raw scores"
        x_labels, y_labels = [], []
        for i, x in enumerate(df.columns):
            x_labels.append(f"#{i} {x}")
            y_labels.append(f"#{i}")
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_yticklabels(y_labels, rotation=0)
        # ax.set_yticklabels(
        #     ax.get_yticklabels(), rotation=45, 
        #     ha="right", va="center", rotation_mode="anchor"
        # )
    ax.set(xlabel="", ylabel="")

    return ax


def main():
    st.title("Platform Data Summary")
    st.sidebar.title("Control panel")

    ## Load data
    data_path = st.sidebar.file_uploader("Custom data", type="csv")
    if data_path is not None:
        df = load_data(data_path)
    else:
        assert DEFAULT_DATA_PATH is not None, "No CSV file found in `data` directory, please upload one."
        df = load_data()
    st.write(f"From: {data_path or DEFAULT_DATA_PATH}")

    ## Print raw data
    st.sidebar.checkbox("Show raw data", value=False, key="raw_data")
    if st.session_state.raw_data:
        st.subheader("Raw data")
        st.write(df.head())

    ## Print descriptive statistics
    st.sidebar.checkbox("Show descriptive statistics", value=True, key="desc_stats")
    if st.session_state.desc_stats:
        st.subheader("Descriptive statistics")
        st.write(
            df.loc[:, ["Age"] + DOMAINS]
            .describe().loc[["count", "min", "50%", "mean", "max", "std"]]
        )

    ## Configure selectable y-axis options
    all_scores = df.columns
    for col in ["SID", "Session", "Age", "Brain Age"]:
        try:
            all_scores = all_scores.drop(col)
        except KeyError:
            pass
    raw_scores = all_scores.drop(DOMAINS)

    st.sidebar.checkbox("Show all scores", value=False, key="all_scores")
    if st.session_state.all_scores:
        y_axis_options = all_scores
    else:
        y_axis_options = DOMAINS

    ## Plot cross-sectional trend
    st.subheader("Cross-sectional plot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        y_axis = st.selectbox("Y-axis", y_axis_options, width=200, key="y_axis_cross_sectional")
    with c2:
        session = st.selectbox("Session", ["All", "1", "2", "3"], width=200, key="session_cross_sectional")
    with c3:
        age_min = st.number_input("Age min", value=None, min_value=None, max_value=None, step=1, width=200, key="age_min_cross_sectional")
    with c4:
        age_max = st.number_input("Age max", value=None, min_value=None, max_value=None, step=1, width=200, key="age_max_cross_sectional")
    st.pyplot(plot_cross_sectional(df, y_axis, session, age_min, age_max).figure)

    ## Plot longitudinal trend
    st.subheader("Longitudinal plot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        y_axis = st.selectbox("Y-axis", y_axis_options, width=200, key="y_axis_longitudinal")
    with c2:
        sid = st.selectbox("Participant", ["All"] + df["SID"].unique().tolist(), width=200)
    with c3:
        age_min = st.number_input("Age min", value=None, min_value=None, max_value=None, step=1, width=200, key="age_min_longitudinal")
    with c4:
        age_max = st.number_input("Age max", value=None, min_value=None, max_value=None, step=1, width=200, key="age_max_longitudinal")
    st.pyplot(plot_longitudinal(df, y_axis, sid, age_min, age_max).figure)

    ## Plot correlation heatmap
    st.subheader("Correlation Heatmap")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        col_type = st.selectbox("Columns", ["domain scores", "raw scores"], width=200)
        if col_type == "domain scores":
            df_sel = df.loc[:, DOMAINS]
        elif col_type == "raw scores":
            df_sel = df.loc[:, raw_scores]
    with c2:
        session = st.selectbox("Session", ["All", "1", "2", "3"], width=200, key="session_corr_heatmap")
    with c3:
        age_min = st.number_input("Age min", value=None, min_value=None, max_value=None, step=1, width=200, key="age_min_corr_heatmap")
    with c4:
        age_max = st.number_input("Age max", value=None, min_value=None, max_value=None, step=1, width=200, key="age_max_corr_heatmap")
    st.pyplot(plot_corr_heatmap(df_sel, col_type, session, age_min, age_max).figure)


if __name__ == "__main__":
    main()
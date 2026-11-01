#!/usr/bin/env python


import itertools
from dateutil.relativedelta import relativedelta
from functools import reduce
from packaging.version import Version
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.colors as pc
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from constants import *


DATA_DIR = Path(__file__).parent / "data"
COLORS = pc.qualitative.Dark2 
THEME = "presentation"
INFO_COLS = ["SID", "Age", "AgeGroup", "Sex", "EduLevel"]
PG_IS_OLD = Version(pg.__version__) < Version("0.7")


## Data ----------------------------------------------------------------------------

def glob_data_paths():
    def _find_latest(pattern):
        files = DATA_DIR.glob(pattern)
        return max(files, key=lambda f: f.stat().st_mtime, default=None)

    return {
        "plat" : _find_latest("tidy_predicted_results*.csv"),  # platform data
        "ques" : _find_latest("Qdata*.xlsx"),  # questionnaire data
        "regis": _find_latest("NHRI*.csv"),  # registration data
        "merged": _find_latest("merged_data*.csv")
    }


@st.cache_data
def load_merged_data(file_paths: dict[str, str | Path]):
    def _load_platform_data(fp: str | Path):
        df = pd.read_csv(fp)
        df.rename(columns=P_DOMAINS_MAP, inplace=True)
        df[P_DOMAINS] = df[P_DOMAINS].astype(float).replace(-1, np.nan)
        df.drop(columns=P_DROP_COLS, inplace=True, errors="ignore")
        df["Date"] = pd.to_datetime(  # e.g., "2025-06-19T024354" or "2026-05-04T021834.472Z"
            df["Date"].str.split("T", expand=True)[0], format="%Y-%m-%d", errors="coerce"
        )
        df["Session"] = df["Session"].astype(int)
        df.sort_values(["SID", "Session"], inplace=True, ignore_index=True)
        return df

    def _load_questionnaire_data(fp: str | Path):
        df = pd.read_excel(fp, usecols=lambda c: str(c).startswith(tuple(Q_USECOLS)))
        q_colname_map = {}
        for k, v in Q_COLNAME_MAP.items():
            col = next((c for c in df.columns if str(c).startswith(k)), None)
            if col is None:
                raise ValueError(f"No column starting with '{k}'")
            q_colname_map[col] = v
        df.rename(columns=q_colname_map, inplace=True)
        
        df["ID8"] = pd.to_numeric(df["ID8"], errors="coerce")
        df.dropna(subset=["ID8"], inplace=True)
        df["SID"] = (
            df["ID8"].astype(int).astype(str).str.zfill(8)
            .str.replace(r"^(\d{4})(\d{4})$", r"\1s\2", regex=True)
        )
        df["Sex"] = df["Sex"].map(Q_SEX_MAP)
        df["IsFemale"] = (df["Sex"] == "F").astype(int).where(df["Sex"].notna())

        df["Edu"] = df["Edu"].where(df["Edu"].isin(list(Q_EDU_MAP)))  # -9999 = missing
        df[["EduYears", "EduLevel"]] = pd.DataFrame(
            df["Edu"].map(Q_EDU_MAP).tolist(), index=df.index
        )
        df["EduGroup"] = df["Edu"].map(Q_EDU_GROUP_MAP)
        return df.loc[:, Q_FINAL_COLS]

    def _load_registration_data(fp: str | Path):
        df = pd.read_csv(fp, usecols=R_COLNAME_MAP)
        df.rename(columns=R_COLNAME_MAP, inplace=True)
        df["BirthDay"] = pd.to_datetime(  # e.g., "06/19/1965"
            df["BirthDay"], format="%m/%d/%Y", errors="coerce"
        )
        df[["SID", "Session"]] = df["ID"].str.split("-", expand=True)
        df["Session"] = df["Session"].astype(int)
        return df.drop(columns="ID")

    if file_paths["merged"] is not None:
        return pd.read_csv(
            file_paths["merged"], 
            parse_dates=["Date", "FirstDate", "BirthDay"]
        )

    df_plat = _load_platform_data(file_paths["plat"])
    df_ques = _load_questionnaire_data(file_paths["ques"])
    df_regis = _load_registration_data(file_paths["regis"])

    df = reduce(
        lambda left, right: pd.merge(left, right, on=["SID", "Session"], how="left"), 
        [df_plat, df_ques, df_regis]
    )

    df["FirstDate"] = df.groupby("SID")["Date"].transform("first")
    ages = [
        relativedelta(date, birthday)
        for date, birthday in zip(df["Date"], df["BirthDay"])
    ]
    df["Age"] = [ age.years for age in ages ]
    df["Age_M"] = [ (age.years * 12) + age.months + (age.days / 30) for age in ages ]
    df["Age_W"] = (df["Date"] - df["BirthDay"]).dt.days / 7
    df["Age_D"] = (df["Date"] - df["BirthDay"]).dt.days

    df = df.assign(**{ 
        c: df.groupby("SID")[c].transform("first") 
        for c in list(set(Q_FINAL_COLS) - {"SID", "Session"})
    })  # in case change across sessions / missing in later sessions

    return df


def make_derived_cols(df: pd.DataFrame, age_unit: str, age_cuts: list[int]):
    if age_unit == "year":
        df["FirstAge"] = df.groupby("SID")["Age"].transform("first")
        df["DateDiff"] = df.apply(lambda row: relativedelta(row["Date"], row["FirstDate"]).years, axis=1)
    elif age_unit == "month":
        df["FirstAge"] = df.groupby("SID")["Age_M"].transform("first")
        diffs = [
            relativedelta(end, start)
            for end, start in zip(df["Date"], df["FirstDate"])
        ]
        df["DateDiff"] = [ (d.years * 12) + d.months + (d.days / 30) for d in diffs ]
    elif age_unit == "week":
        df["FirstAge"] = df.groupby("SID")["Age_W"].transform("first")
        df["DateDiff"] = (df["Date"] - df["FirstDate"]).dt.days / 7
    else:  # day
        df["FirstAge"] = df.groupby("SID")["Age_D"].transform("first")
        df["DateDiff"] = (df["Date"] - df["FirstDate"]).dt.days

    df["AgeGroup"] = pd.cut(
        df["Age"], [-np.inf, *age_cuts, np.inf], right=False, 
        labels=(
            [f"< {age_cuts[0]}"] + 
            [f"{a}-{b - 1}" for a, b in zip(age_cuts, age_cuts[1:])] + 
            [f"{age_cuts[-1]}+"]
        )
    )

    return df


def filter_age(df: pd.DataFrame, age_min: int | None, age_max: int | None):
    if age_min is not None:
        df = df.query(f"Age >= {age_min}")
    if age_max is not None:
        df = df.query(f"Age <= {age_max}")

    return df


def select_session(df: pd.DataFrame, session: str):
    if session == "mean":
        df = df.assign(**{ 
           c: df.groupby("SID")[c].transform("first") 
            for c in INFO_COLS
        })  # in case change in different sessions for the same person
        return (
            df.groupby(INFO_COLS, as_index=False, observed=True)
            .mean(numeric_only=True)
        )
    # elif session == "all":
    #     return df
    else:
        return df.query(f"Session == {session}")


## Statistics ----------------------------------------------------------------------

@st.cache_data
def fit_lmem(df: pd.DataFrame, y_col: str, covars: list[str] = []):
    terms = ["FirstAge", "DateDiff", *covars]
    df = df.loc[:, ["SID", y_col, *terms]].dropna()
    if df["SID"].nunique() < 5:
        return None, None, "Too few participants"

    formula = f"{y_col} ~ {' + '.join(terms)}"
    try:
        res = smf.mixedlm(formula, df, groups="SID", re_formula="~DateDiff").fit()
        random_effects = "(1 + DateDiff | SID)"
    except (np.linalg.LinAlgError, ValueError):  # the random slope cannot be estimated
        res = smf.mixedlm(formula, df, groups="SID").fit()
        random_effects = "(1 | SID)"

    ci = res.conf_int()
    coefs = pd.DataFrame({
        "Coef": res.fe_params, 
        "CI low": ci.loc[res.fe_params.index, 0], 
        "CI high": ci.loc[res.fe_params.index, 1],
        "p": res.pvalues[res.fe_params.index],
    }).round(3)

    full_formula = f"{formula} + {random_effects}"
    notes = f"{df['SID'].nunique()} participants, {len(df)} observations"
    if not res.converged:
        notes += "; did not converge"

    return coefs, full_formula, notes


@st.cache_data
def calc_reliability(
    df_wide_dict: dict[str, pd.DataFrame], 
    sessions: list[int], 
    show_icc1: bool, 
    show_icca: bool, 
    icc_thres: float = 0.75, 
    p_thres: float = 0.05
):
    def _sel_icc_types():
        new_icc_names = np.array([
            "ICC(1,1)", "ICC(A,1)", "ICC(C,1)", "ICC(1,k)", "ICC(A,k)", "ICC(C,k)"
        ])
        old_icc_names = np.array([
            "ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"
        ])
        idxs = sorted(
            [2, 5] + ([0, 3] if show_icc1 else []) + ([1, 4] if show_icca else [])
        )
        return new_icc_names[idxs], old_icc_names[idxs]

    ci_col = "CI95%" if PG_IS_OLD else "CI95"
    new_icc_names, old_icc_names = _sel_icc_types()

    rows = []
    bold_masks = []

    for domain in P_DOMAINS:
        df_wide = df_wide_dict[domain]
        row = {"Domain": domain, "N obs.": len(df_wide)}
        bold_mask = {}

        if len(df_wide) < 5:
            rows.append(row)
            bold_masks.append(bold_mask)
            continue

        df_long = ( 
            df_wide.melt(
                var_name="Session", 
                value_name=domain, 
                ignore_index=False  # keep "SID" as index
            ).reset_index()  # convert "SID" to a column
        )
        res = pg.intraclass_corr(
            data=df_long, 
            targets="SID", 
            raters="Session", 
            ratings=domain
        )
        if PG_IS_OLD:
            res["Type"] = res["Type"].map(dict(zip(old_icc_names, new_icc_names)))
        res.set_index("Type", inplace=True)
        
        for icc_type in new_icc_names:
            icc = res.loc[icc_type, "ICC"]
            ci = res.loc[icc_type, ci_col]
            row[icc_type] = f"{icc:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
            bold_mask[icc_type] = (icc >= icc_thres)

        for ses in sessions:
            row[f"S{ses}"] = df_wide[ses].mean()

        if len(sessions) == 2:
            s1, s2 = sessions[0], sessions[1]
            tt = stats.ttest_rel(  # t-test on rwo related samples of scores
                df_wide[s2], df_wide[s1], alternative="two-sided"
            )
            stats_p = tt.pvalue
            stats_cols = {
                f"S{s2} - S{s1}": (df_wide[s2] - df_wide[s1]).mean(),
                "t": tt.statistic,
                "p(t)": stats_p, 
            }
        else:
            aov = pg.rm_anova(  # two-way repeated measures ANOVA
                data=df_long, dv=domain, within="Session", subject="SID", detailed=False
            )
            p_col = next(c for c in ["pval", "p-val", "p_val", "p-unc", "p_unc"] if c in aov.columns)
            stats_p = aov.loc[0, p_col]
            stats_cols = {
                "F": aov.loc[0, "F"], 
                "p(F)": stats_p, 
            }

        row.update(stats_cols)
        bold_mask.update({ x: stats_p < p_thres for x in stats_cols.keys() })

        rows.append(row)
        bold_masks.append(bold_mask)

    fmt_dict = { f"S{s}": "{:.1f}" for s in sessions }
    stats_fmt = ["{:.1f}", "{:.2f}", "{:.3f}"] if len(sessions) == 2 else ["{:.2f}", "{:.3f}"]
    fmt_dict.update({ k: v for k, v in zip(stats_cols.keys(), stats_fmt) })

    notes = f"ICC above {icc_thres} and stats with p < {str(p_thres).lstrip('0')} are marked as bold."

    return pd.DataFrame(rows), bold_masks, fmt_dict, notes


@st.cache_data
def calc_correlations(df: pd.DataFrame, cols: list[str], method: str, controls: list[str], is_within: bool):
    def _corr(x, y):
        targ_df = df[["SID", x, y, *controls]].dropna()
        try:
            if is_within:
                res = pg.rm_corr(targ_df, x=x, y=y, subject="SID")
            elif len(controls) > 0:
                res = pg.partial_corr(targ_df, x=x, y=y, covar=controls, method=method)
            else:
                res = pg.corr(targ_df[x], targ_df[y], method=method)

        except (AssertionError, ValueError, np.linalg.LinAlgError):  # too few rows left by the filters
            return {}

        return (
            res.iloc[0].to_dict() if not PG_IS_OLD else
            res.rename(columns={"pval": "p_val", "p-val": "p_val", "CI95%": "CI95"}).iloc[0].to_dict()
        )

    return pd.DataFrame([ 
        {"X": x, "Y": y, **_corr(x, y)} 
        for x, y in itertools.combinations(cols, 2) 
    ])


@st.cache_data
def calc_corr_pc1(corr_dfs: dict[str, pd.DataFrame], cols: list[str]):
    rows = []

    for g, df in corr_dfs.items():
        if "r" not in df.columns or df["r"].isna().all():
            continue

        n_cols = len(cols)
        R = pd.DataFrame(np.eye(n_cols), index=cols, columns=cols)
        for _, row in df.iterrows():
            x, y, r = row["X"], row["Y"], row["r"]
            if (x not in cols) or (y not in cols):
                continue
            if pd.notna(r):
                R.loc[x, y] = r
                R.loc[y, x] = r

        cormat = R.fillna(0.0).to_numpy(float)
        eigvals, eigvecs = np.linalg.eigh(cormat)

        lambda_1 = eigvals[-1]
        v_1 = eigvecs[:, -1]
        pve = (lambda_1 / n_cols) * 100  # proportion of variation explained
        # loading_1 = v_1 * np.sqrt(max(lambda_1, 0))  # correlation coefficientof each factor with PC1

        rows.append({
            "AgeGroup": str(g), 
            "PC1 %": pve
        })

    return pd.DataFrame(rows)


@st.cache_data
def run_clust_and_proj(
    df: pd.DataFrame, 
    _scaler: StandardScaler, 
    cols: list[str], 
    clust_method: str, 
    proj_method: str, 
    k: int, 
    seed: int = 0
):
    # tell Streamlit not to hash the StandardScaler (nnhashable) by adding a leading underscore to the argument's name

    df_ok = df.dropna(subset=cols)
    if len(df_ok) < 2 * k:  # too few participants left by the filters
        return None, None

    df_z = _scaler.transform(df_ok[cols])
    if clust_method == "K-Means":
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(df_z)
        centers = model.cluster_centers_  # ndarray of shape (k, cols)
    else:  # "Gaussian mixture"
        model = GaussianMixture(n_components=k, random_state=seed)
        labels = model.fit_predict(df_z)
        centers = model.means_

    if proj_method == "PCA":
        reducer = PCA(n_components=2, random_state=seed)
    else:  # "t-SNE"
        reducer = TSNE(n_components=2, random_state=seed)

    Z_2d = reducer.fit_transform(df_z)
    C_2d = reducer.transform(centers)

    df_2d = (
        df_ok.loc[:, INFO_COLS]
        .assign(Cluster=labels + 1, X=Z_2d[:, 0], Y=Z_2d[:, 1])
    )
    df_centers = pd.DataFrame(data=C_2d, columns=["X", "Y"], index=range(1, k+1))
    
    return df_2d, df_centers


## Plots ---------------------------------------------------------------------------

def plot_histograms(
    df: pd.DataFrame, 
    shared_yaxes: bool, 
    bin_size: int, 
    pad: int = 2
):
    fig = make_subplots(
        rows=1,
        cols=len(P_DOMAINS),
        shared_yaxes=shared_yaxes,
        subplot_titles=P_DOMAINS,
    )

    for i, col in enumerate(P_DOMAINS, start=1):
        fig.add_trace(
            go.Histogram(
                x=df[col], 
                xbins=dict(start=0, end=100 + bin_size, size=bin_size), 
                marker_color=COLORS[0], 
                showlegend=False,
            ), 
            row=1, col=i
        )

    fig.update_xaxes(range=[0, 100 + pad])
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0.05, height=200, template=THEME
    )

    return fig


def plot_cross_sectional(
    df: pd.DataFrame, 
    y_cols: list[str], 
    fit_type: str, 
    shared_yaxes: bool, 
    pad: int = 2
):
    def _add_fit(x: pd.Series, y: pd.Series, **pos):
        ok = x.notna() & y.notna()
        x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)
        if len(x) < 5 or np.ptp(x) == 0:  # ptp: peak-to-peak range
            return

        x_new = np.linspace(x.min(), x.max(), 100)  # predict values at which X points
        
        if fit_type == "LOWESS":  # Locally Weighted Scatterplot Smoothing
            y_fit = sm.nonparametric.lowess(y, x, xvals=x_new)
            kwargs = {"hoverinfo": "skip"}

        else:  # polynomial with a 95% CI band
            order = 2 if fit_type == "Quadratic" else 1
            X = np.vander(x - x.mean(), order + 1)  # Vandermonde matrix
            mdl = sm.OLS(y, X).fit()  # fitted model
            X_new = np.vander(x_new - x.mean(), order + 1)
            pred = mdl.get_prediction(X_new).summary_frame()
            fig.add_trace(
                go.Scatter(
                    x=np.r_[x_new, x_new[::-1]], 
                    y=np.r_[pred["mean_ci_upper"], pred["mean_ci_lower"][::-1]],
                    fill="toself", fillcolor="rgba(0, 0, 0, 0.15)", 
                    line_width=0, hoverinfo="skip", showlegend=False,
                ), 
                **pos
            )
            c = mdl.params
            kwargs = {"hovertemplate": (
                f"y = {c[0]:.1f}*x^2 + {c[1]:.1f}*x + {c[2]:.1f}<extra></extra>" 
                if fit_type == "Quadratic" else 
                f"y = {c[0]:.1f}*x + {c[1]:.1f}<extra></extra>"
            )}

        fig.add_trace(
            go.Scatter(
                x=x_new, 
                y=y_fit if fit_type == "LOWESS" else pred["mean"], 
                mode="lines", 
                line=dict(color="black", width=2), 
                showlegend=False,
                **kwargs
            ), 
            **pos
        )

    fig = make_subplots(
        rows=1, 
        cols=len(y_cols), 
        shared_yaxes=shared_yaxes, 
        horizontal_spacing=0.015, 
        subplot_titles=y_cols if len(y_cols) > 1 else None
    )

    for i, c in enumerate(y_cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["Age"], y=df[c], mode="markers", 
                marker=dict(color=COLORS[0], size=7, opacity=0.5),
                showlegend=False, 
                customdata=df.loc[:, ["SID", "Session", "Age"]], 
                hovertemplate=f"SID: %{{customdata[0]}}<br>Session: %{{customdata[1]}}<br>Age: %{{customdata[2]}}<br>{c}: %{{y}}<extra></extra>",
            ), 
            row=1, col=i
        )
        _add_fit(df["Age"], df[c], row=1, col=i)

    fig.update_xaxes(title_text="Age", range=[df["Age"].min() - pad, df["Age"].max() + pad])
    kwargs = {"range": [0, 100 + pad]} if all(c in P_DOMAINS for c in y_cols) else {}
    fig.update_yaxes(title_text="Score", row=1, col=1, **kwargs)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=10),
        height=350, template=THEME
    )

    return fig


def plot_longitudinal(
    df: pd.DataFrame, 
    y_col: str, 
    x_type: str, 
    pad: int = 5
):  
    x = df["FirstAge"] + df["DateDiff"] if x_type == "Age" else df["Session"]

    color_cycle = itertools.cycle(COLORS)
    fig = go.Figure()

    for sid, sub_df in df.assign(x=x).groupby("SID"):
        color = next(color_cycle)
        fig.add_trace(
            go.Scatter(
                x=sub_df["x"], 
                y=sub_df[y_col], 
                mode="lines+markers",
                name=str(sid),
                line=dict(color=color, width=1.5),
                marker=dict(color=color, size=7, opacity=0.8),
                opacity=0.8,
                showlegend=False, 
                customdata=sub_df.loc[:, ["SID", "Session", "Age"]],
                hovertemplate=f"SID: %{{customdata[0]}}<br>Session: %{{customdata[1]}}<br>Age: %{{customdata[2]}}<br>{y_col}: %{{y}}<extra></extra>",
            )
        )

    if x_type == "Age":
        age_unit = st.session_state["age_unit"]
        fig.update_xaxes(title_text=f"Age in {age_unit} (FirstAge + DateDiff)")
    else:
        fig.update_xaxes(title_text="Session", dtick=1)

    fig.update_yaxes(title_text=y_col, range=[0, 100 + pad] if y_col in P_DOMAINS else None)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=40),
        height=400, template=THEME
    )

    return fig


def plot_reliability(df_wide: pd.DataFrame, y_col: str):
    pairs = list(itertools.combinations([1, 2, 3], 2))
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, 
        subplot_titles=[ f"{s1} vs. {s2}" for s1, s2 in pairs ]
    )

    for i, (s1, s2) in enumerate(pairs, start=1):
        mean = (df_wide[s1] + df_wide[s2]) / 2
        diff = df_wide[s2] - df_wide[s1]
        bias, sd = diff.mean(), diff.std()

        fig.add_trace(
            go.Scatter(
                x=mean, y=diff, mode="markers", 
                marker=dict(color=COLORS[0], size=7, opacity=0.7), 
                customdata=df_wide.index,
                hovertemplate="SID: %{customdata}<br>mean: %{x}<br>diff: %{y}<extra></extra>", 
                showlegend=False,
            ), 
            row=1, col=i
        )

        reg = stats.linregress(mean, diff)
        x_line = np.array([mean.min(), mean.max()])
        y_line = reg.slope * x_line + reg.intercept
        p_text = "<i>p</i> < .001" if reg.pvalue < .001 else f"<i>p</i> = {f'{reg.pvalue:.3f}'.lstrip('0')}"
        p_text += "".join([ "*" if reg.pvalue < t else "" for t in [.05, .01, .001] ])
        stats_text = f"<i>r</i> = {reg.rvalue:.2f} ({p_text})"

        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode="lines", 
                line=dict(
                    color="forestgreen" if reg.pvalue < 0.05 else "black", 
                    dash="solid" if reg.pvalue < 0.05 else "dashdot"
                ), 
                hovertemplate=f"Slope = {reg.slope:.2f}<br>{stats_text}<extra></extra>", 
                showlegend=False,
            ), 
            row=1, col=i
        )
        fig.add_annotation(
            text=f"Oldham {stats_text}", 
            xref=f"x{'' if i == 1 else i} domain", xanchor="center", x=0.5, 
            yref=f"paper", yanchor="top", y=-0.2, 
            showarrow=False
        )
        
        for value, label, dash_style, color in [
            (bias, "bias", "solid", "blue"),
            (bias + 1.96 * sd, "+1.96 SD", "dash", "red"),
            (bias - 1.96 * sd, "-1.96 SD", "dash", "red"), 
            (0, "zero", "dot", "gray")
        ]:
            fig.add_hline(
                y=value, line_dash=dash_style, line_color=color,
                annotation_text=("" if label == "zero" else f"{label} = {value:.1f}"), 
                annotation_font_color=color, 
                annotation_position=("bottom right" if label == "zero" else "top right"), 
                row=1, col=i
            )

        fig.update_xaxes(title_text=f"mean(S{s1}, S{s2})", row=1, col=i)
        fig.update_yaxes(title_text=f"S{s2} - S{s1}", row=1, col=i)
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=80),
        height=400, template=THEME
    )

    return fig


def plot_corr_heatmap(
    corr_dfs: dict[str, pd.DataFrame], 
    cols: list[str], 
    show_scale: bool = False
):
    def _cormat(df: pd.DataFrame, val: str):
        return (
            df.pivot(index="Y", columns="X", values=val)
            .reindex(index=y_cols, columns=x_cols)
        )

    with_moca = ("MoCA" in cols)
    x_cols, y_cols = cols[:-1], cols[1:]
    only_domain_cols = set(cols).issubset(set(P_DOMAINS + ["MoCA"]))

    groups = [
        g for g, df in corr_dfs.items()
        if "r" in df.columns and not df["r"].isna().all()
    ]
    n_groups = len(groups)

    box_size = 300 if only_domain_cols else 850
    margin_l = 90 if only_domain_cols else 60
    margin_r = 40
    margin_t = 50
    margin_b = 50 if only_domain_cols else 160
    gap_px = 80 if only_domain_cols else 60
    fw = box_size + margin_l + margin_r
    fh = (box_size * n_groups) + (gap_px * (n_groups - 1)) + margin_t + margin_b
    plot_area_h = fh - margin_t - margin_b
    v_space = (gap_px / plot_area_h) if n_groups > 1 else 0.0

    fig = make_subplots(
        rows=n_groups, cols=1, shared_xaxes=(not only_domain_cols), 
        vertical_spacing=v_space, subplot_titles=groups
    )

    for i, (g, df) in enumerate(corr_dfs.items(), start=1):
        if g not in groups:
            continue

        df["r_sig"] = df.apply(
            lambda row: (
                ( f"{row['r']:.2f}" if only_domain_cols else "" ) + 
                "".join([ "*" if row["p_val"] < t else "" for t in [.05, .01, .001] ])
            ), 
            axis=1
        )        
        df["hover"] = df.apply(
            lambda row: "Insufficient data" if (
                pd.isna(row["r"]) 
                or not isinstance(row.get("CI95"), (list, tuple, np.ndarray))
            ) else (
                ( f"{row['Y']} × {row['X']}<br>" if only_domain_cols  
                else f"{row['Y']} × <br>{row['X']}<br>" ) + 
                ( f"n = {row['n']}<br>" if "n" in df.columns else f"dof = {row['dof']}<br>" ) + 
                f"95% CI = [{row['CI95'][0]:.2f}, {row['CI95'][1]:.2f}]<br>" + 
                f"<i>p</i> = {row['p_val']:.3g}<br>" + 
                ( f"power = {row['power']:.3f}" if "power" in df.columns else "" )
            ), 
            axis=1
        )

        fig.add_trace(
            go.Heatmap(
                z=_cormat(df, "r").to_numpy(float), x=x_cols, y=y_cols, 
                zmin=-1, zmax=1, colorscale="RdBu_r", 
                text=_cormat(df, "r_sig").fillna("").to_numpy(), 
                texttemplate="%{text}", # textfont={"size": 16 if only_domain_cols else 8}, 
                hovertext=_cormat(df, "hover").fillna("").to_numpy(), 
                hoverinfo="text", 
                showscale=show_scale
            ), 
            row=i, col=1
        )
        if with_moca:
            fig.add_vline(
                x=0.5, line_width=3, line_color="white", 
                row=i, col=1
            )
        
        fig.update_yaxes(
            scaleanchor="x" if i == 1 else f"x{i}", 
            scaleratio=1, autorange="reversed", 
            row=i, col=1
        )

    if not only_domain_cols:
        fig.update_xaxes(tickvals=cols, ticktext=[f"#{i} {x}" for i, x in enumerate(cols)])
        fig.update_yaxes(tickvals=cols, ticktext=[f"#{i}" for i in range(len(cols))])

    fig.update_layout(
        height=fh, width=fw,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b), 
        template=THEME
    )

    return fig 


def plot_pc1_bars(df: pd.DataFrame):
    fig = go.Figure(
        go.Bar(x=df["AgeGroup"], y=df["PC1 %"], marker_color=COLORS[0])
    )    
    fig.update_layout(
        yaxis_title="% variance explained", 
        height=300, width=100 + (50 * len(df)), 
        margin=dict(l=40, r=40, t=0, b=50),
        template=THEME
    )

    return fig


def plot_radar(df: pd.DataFrame, scaler: StandardScaler, sid: str, cols: list[str]):
    df_z = scaler.transform(df[cols])
    sidx = df["SID"].to_list().index(sid)
    g = df["AgeGroup"].iloc[sidx]
    g_mask = (df["AgeGroup"] == g).to_numpy()

    fig = go.Figure(
        go.Scatterpolar(
            r=np.nanmean(df_z[g_mask], axis=0), theta=cols, 
            fill="toself", name="group", # showlegend=False, 
            hovertemplate="Z-score: %{r:.2f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=df_z[sidx], theta=cols, 
            fill="toself", name="participant", # showlegend=False, 
            hovertemplate="Z-score: %{r:.2f}<extra></extra>"
        )
    )
    fig.add_annotation(
        text=(
            "<i>※ Participant's info.</i><br>" +
            f"Age: {df['Age'].iloc[sidx]}<br>" +
            f"AgeGroup: {g} (n = {sum(g_mask)})<br>" +
            f"Gender: {df['Sex'].iloc[sidx]}<br>" +
            f"EduLevel: {df['EduLevel'].iloc[sidx]}"
        ),  
        xref=f"paper", xanchor="left", x=0, 
        yref=f"paper", yanchor="top", y=0.98, 
        align="left", showarrow=False
    )
    fig.update_layout(
        # title=dict(text="Z-scores", x=0.5, y=0.95), 
        polar=dict(
            radialaxis=dict(visible=True), 
            domain=dict(x=[0.1, 1], y=[0.1, 1])
        ), 
        legend=dict(y=0.3, yanchor="bottom"), 
        margin=dict(l=20, r=0, t=0, b=10),
        height=600, template=THEME
    )

    return fig


def plot_clusters(df_2d: pd.DataFrame, df_centers: pd.DataFrame, proj_method: str, sid: str | None):
    clusters = sorted(df_2d["Cluster"].unique())
    x_label, y_label = ("PC1", "PC2") if proj_method == "PCA" else ("t-SNE 1", "t-SNE 2")
    fig = go.Figure()

    for i, cid in enumerate(clusters):
        sub_df = df_2d.query(f"Cluster == {cid}")
        fig.add_trace(
            go.Scatter(
                x=sub_df["X"], y=sub_df["Y"], mode="markers", 
                marker=dict(
                    color=COLORS[i % len(COLORS)], size=7, opacity=0.8, 
                    line=dict(width=0.5, color="white")
                ),
                name=f"Cluster {cid} (n = {len(sub_df)})", legendgroup=str(cid), 
                customdata=sub_df.loc[:, ["SID", "Age", "Sex"]], 
                hovertemplate="SID: %{customdata[0]}<br>Age: %{customdata[1]}<br>Sex: %{customdata[2]}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[ df_centers.at[cid, "X"] ], y=[ df_centers.at[cid, "Y"] ], mode="markers", 
                marker=dict(
                    symbol="x", color=COLORS[i % len(COLORS)], size=13, 
                    line=dict(width=2, color="white")
                ),
                name=f"#{cid} center", legendgroup=str(cid), showlegend=False, 
            )
        )

    if sid is not None:
        sidx = df_2d["SID"].to_list().index(sid)
        df_2d.set_index("SID", inplace=True)
        fig.add_trace(
            go.Scatter(
                x=[ df_2d["X"].iloc[sidx] ], y=[ df_2d["Y"].iloc[sidx] ], mode="markers", 
                marker=dict(
                    color="rgba(0,0,0,0)", size=8, 
                    line=dict(width=2, color="black")
                ),
                name=sid, showlegend=False, hoverinfo="skip" 
            )
        )

    grid_style = dict(
        tick0=0, dtick=1, griddash="dash", 
        zeroline=True, # zerolinecolor="lightgray"
    )
    fig.update_xaxes(title_text=x_label, **grid_style)
    fig.update_yaxes(title_text=y_label, **grid_style)
    fig.update_layout(
        legend=dict(
            orientation="v", yanchor="top", y=1.1, xanchor="left", x=0, 
            traceorder="grouped", tracegroupgap=2
        ), 
        margin=dict(l=0, r=0, t=40, b=0),
        height=450, template=THEME,
    )

    return fig


def custom_red_map(val, vmin, vmax):
    if val == vmin:
        return ""  # maintain transparent background
        # return "background-color: #ffffff; color: black;"

    ratio = 0.15 + 0.85 * min(val / vmax, 1.0)
    bg_color = pc.sample_colorscale("Reds", [ratio])[0]
    text_color = "white" if ratio > 0.6 else "black"

    return f"background-color: {bg_color}; color: {text_color};"


## Sidebar and Tabs ----------------------------------------------------------------

def setup_data_sidebar():
    def _setup_data_paths():
        data_paths = glob_data_paths()  # default paths

        with st.sidebar:
            with st.expander("Upload", expanded=True):
                fp_0 = st.file_uploader("Merged data", type="csv")
                fp_1 = st.file_uploader("Platform data", type="csv")
                fp_2 = st.file_uploader("Questionnaire data", type="xlsx")
                fp_3 = st.file_uploader("Registration data", type="csv")

        for k, fp in zip(["merged", "plat", "ques", "regis"], [fp_0, fp_1, fp_2, fp_3]):
            if fp is not None:
                data_paths[k] = fp  # update if uploaded

        if (data_paths["merged"] is None):
            if any(data_paths[k] is None for k in ["plat", "ques", "regis"]):
                st.info("No local data found, please upload one.")
                st.stop()  # avoid error

        return data_paths
                
    def _setup_and_derive_cols(df: pd.DataFrame):
        st.sidebar.subheader("Options")
        age_unit_options = ["year", "month", "week", "day"]
        default_idx = age_unit_options.index("month")
        st.sidebar.selectbox("Longitudinal age unit", age_unit_options, index=default_idx, key="age_unit")
        age_cuts = st.sidebar.multiselect("Age group cut points", list(range(61, 86)), default=[70, 80])
        
        return make_derived_cols(df, st.session_state["age_unit"], sorted(age_cuts))

    def _setup_and_apply_filters(df: pd.DataFrame):
        with st.sidebar:
            st.subheader("Filters")
            session = st.selectbox("Session", ["mean", "1", "2", "3"])
            c1, c2 = st.columns(2)
            with c1:
                age_min = st.number_input("Age min (yr)", value=None, step=1)
            with c2:
                age_max = st.number_input("Age max (yr)", value=None, step=1)

        df_filt1 = filter_age(df, age_min, age_max)
        if len(df_filt1) == 0:
            st.info("No data passed the age filter.")
            st.stop()

        df_filt2 = select_session(df_filt1, session) 
        if len(df_filt2) == 0:
            st.info("No data passed the session selection.")
            st.stop()

        return df_filt2, df_filt1

    st.sidebar.title("Data panel")
    data_paths = _setup_data_paths()
    df = load_merged_data(data_paths)
    st.sidebar.download_button(
        label="Download merged CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="merged_data.csv",
        mime="text/csv",
    )
    df = _setup_and_derive_cols(df)
    df_filt2, df_filt1 = _setup_and_apply_filters(df)

    return df_filt2, df_filt1


def section_overview(df_filt1: pd.DataFrame, df_filt2: pd.DataFrame):
    st.markdown(
        "### Sample composition " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":red-badge[:material/close: session filtered] "
    )
    sep_age_group = st.checkbox("Count separately for different age groups", value=False)
    group_by = "Session"
    if sep_age_group:
        group_by = ["Session", "AgeGroup"]

    ses_total = df_filt1.groupby(group_by)["SID"].nunique().rename("Total")
    by_domain = df_filt1.groupby(group_by)[P_DOMAINS].count()

    show_missing = st.checkbox("Show missing data", value=True)
    if show_missing:
        by_domain = by_domain.rsub(ses_total, axis=0)
        vmax = by_domain.values.max()
        by_domain = by_domain.style.map(lambda x: custom_red_map(x, 0, vmax))
        # by_domain = by_domain.style.background_gradient(cmap="Reds", axis=None, vmin=0)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.dataframe(ses_total)
    with c2:
        st.dataframe(by_domain)

    st.markdown(
        "### Data distributions " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":blue-badge[:material/check: session filtered] "
    )
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        bin_size = st.number_input("Bin size", value=5, step=1, width=200)
    with c2:
        share_y = st.checkbox("Share y-axes", value=False)

    st.plotly_chart(
        plot_histograms(df_filt2, share_y, bin_size)
    )

    st.markdown(
        "### Descriptive statistics " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":blue-badge[:material/check: session filtered] "
    )
    st.dataframe(
        df_filt2.loc[:, ["Age"] + P_DOMAINS]
        .describe().loc[["count", "min", "mean", "50%", "max", "std"]]
        .round(2)
    )

    st.markdown(
        "### Raw data " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":yellow-badge[:material/change_history: session filtered] "
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        sid = st.selectbox("Participant", df_filt1["SID"].unique().tolist(), index=None)
        sid_selected = (sid is not None)
        df_raw = df_filt1.query(f"SID == '{sid}'") if sid_selected else df_filt1
    with c2:
        sex = st.selectbox("Gender", df_filt1["Sex"].unique().tolist(), index=None, disabled=sid_selected)
        df_raw = df_raw.query(f"Sex == '{sex}'") if (sex is not None) else df_raw
    with c3:
        edu = st.selectbox("EduLevel", df_filt1["EduLevel"].unique().tolist(), index=None, disabled=sid_selected)
        df_raw = df_raw.query(f"EduLevel == '{edu}'") if (edu is not None) else df_raw
    with c4:
        ses = st.selectbox("Session", df_filt1["Session"].unique().tolist(), index=None, disabled=sid_selected)
        df_raw = df_raw.query(f"Session == {ses}") if (ses is not None) else df_raw
    with c5:
        main_cols_only = st.checkbox("Main columns only", value=True)
        if main_cols_only:
            df_raw = df_raw.loc[:, INFO_COLS + P_DOMAINS]

    st.dataframe(df_raw.head(), hide_index=True)
    st.caption(f"{df_raw.shape[0]} rows × {df_raw.shape[1]} columns", text_alignment="right")


def section_cross_sectional(df_filt2: pd.DataFrame):
    st.markdown(
        "### Cross-sectional trend " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":blue-badge[:material/check: session filtered] "
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        fit_type = st.selectbox("Fit", ["Linear", "Quadratic", "LOWESS"], width=200)
    with c2:
        show_raw_scores = st.checkbox("Include raw scores", value=False)
    with c3:
        share_y = st.checkbox("Share y-axes", value=True)
    
    y_options = P_DOMAINS + P_RAW_SCORES if show_raw_scores else P_DOMAINS
    y_list = st.multiselect("Y-axes", y_options, default=P_DOMAINS)

    st.plotly_chart(
        plot_cross_sectional(df_filt2, y_list, fit_type, share_y)
    )


def section_longitudinal(df_filt1: pd.DataFrame):
    st.markdown(
        "### Longitudinal trend " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":red-badge[:material/close: session filtered] "
    )
    # st.subheader("Longitudinal trend")
    # st.badge("age filtered", icon=":material/check:", color="blue")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_type = st.selectbox("X-axis", ["Age", "Session"])
    with c2:
        y_col = st.selectbox("Y-axis", P_DOMAINS + P_RAW_SCORES)
    with c3:
        sid = st.selectbox("Participant", ["All"] + df_filt1["SID"].unique().tolist())

    df = df_filt1
    if sid != "All":
        df = df_filt1.query(f"SID == '{sid}'")

    st.plotly_chart(
        plot_longitudinal(df, y_col, x_type)
    )

    st.markdown(
        "### Linear mixed-effect model " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":red-badge[:material/close: session filtered] "
    )
    covars = st.multiselect("Covariates", ["IsFemale", "EduYears", "MoCA"], width=400)
    coefs, formula, notes = fit_lmem(df_filt1, y_col, covars)

    st.latex(formula.replace("~", r"\sim").replace("|", r"\mid"))
    if coefs is not None:
        st.dataframe(coefs)
    st.caption(notes, text_alignment="right")


def section_reliability(df_filt1: pd.DataFrame):
    st.markdown(
        "### Intra-class correlation (ICC) " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":red-badge[:material/close: session filtered] "
    )
    with st.expander("About ICC", expanded=False):
        st.markdown(dedent(r'''
            ICC measures the ratio of between-target variance to total variance. \
            `pg.intraclass_corr()` returns six ICC variants, organized along two dimensions:
            1. **How bias is handled:**
                - `ICC(1,)`: Assumes raters are interchangeable with no systematic bias. 
                             Only valid when rater means are roughly equal.
                - `ICC(A,)`: Absolute agreement. Penalises systematic differences between raters.
                             (*"do raters assign the same values?"*)
                - `ICC(C,)`: Consistency. Ignores systematic differences between raters.
                             (*"do raters rank targets in the same order?"*)
                             Equivalent to the reliability expected if rater biases were removed.
            2. **Single or averaged scores:**
                - `1`: Reliability of one rating by a single rater.
                - `k`: Reliability of the mean across all raters.

            **Practical guidance:** \
            When `ICC(C,1)` is notably larger than `ICC(A,1)`, non-negligible bias is likely present. \
            In that case, `ICC(1,1)` is invalid and should not be reported. \
            The F statistic and p-value in the output provide a formal test of whether rater means differ significantly.

            **General benchmarks for ICC values:**
            ```text
            0.00 -------------------- 0.50 ---------- 0.75 ------ 0.90 ---- 1.00
                         Poor               Moderate        Good       Excellent
            ```
            **See more:** 
            - http://www.real-statistics.com/reliability/intraclass-correlation/
            - https://pingouin-stats.org/generated/pingouin.intraclass_corr.html
            - https://github.com/raphaelvallat/pingouin/issues/485
        '''))

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        sessions = st.multiselect("Sessions to include", [1, 2, 3], default=[1, 2, 3], width=300)
    with c2:
        show_icc1 = st.checkbox("Show ICC(1,)", value=False)
    with c3:
        show_icca = st.checkbox("Show ICC(A,)", value=False)

    df_wide_dict = {
        domain: (
            df_filt1
            .pivot_table(index="SID", columns="Session", values=domain)
            .reindex(columns=sessions)  # fixed column order
            .dropna()
        ) for domain in P_DOMAINS 
    }
    df, bold_masks, fmt_dict, notes = calc_reliability(df_wide_dict, sorted(sessions), show_icc1, show_icca)
    mask_df = pd.DataFrame(bold_masks).reindex(columns=df.columns, fill_value=False)
    df = (
        df.style
        .apply(lambda _: mask_df.map(lambda x: "font-weight: bold" if x else ""), axis=None)
        .format(fmt_dict, na_rep="-")
    )
    st.dataframe(df, hide_index=True)
    st.caption(notes, text_alignment="right")

    st.markdown(
        "### Bland-Altman plot (+ Oldham fit) " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":red-badge[:material/close: session filtered] "
    )
    c1, c2 = st.columns([1, 4], vertical_alignment="bottom")
    with c1:
        domain = st.selectbox("Domain", P_DOMAINS)
    with c2:
        with st.expander("About Bland-Altman plot", expanded=False):
            st.markdown(dedent(r'''
            **The three main horizontal lines in the graph**:
            1. **Mean Difference/Bias:** Represents the average deviation of the measurement results from the two methods, used to observe whether there is systematic bias (*i.e., whether one method is systematically higher or lower than the other*).
            2. **Limits of Agreement (LoA):** Usually set as "mean difference ± 1.96 × standard deviation (SD) of the difference". Approximately 95% of the difference points will fall within this upper and lower limits.
            3. **Zero Difference Line (Reference Line):** Represents the position where there is absolutely no difference between the two methods. The closer the mean difference line is to the zero line, the more consistent the two methods are.
            
            **How to Interpret**:
            - **Consistency Judgment:** Check whether the width of the 95% consistency limit interval is within an acceptable range. If the interval is too wide, it means that the difference between the two methods is too large and they cannot be arbitrarily substituted for each other.
            - **Trends and Bias:** Does the distribution of observation points become skewed or divergent as the X-axis (mean value) changes (*e.g., the larger the value, the greater the difference*)? This indicates that the error is related to the magnitude of the measured value.
            
            **See also**
            - [Wikipedia](https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot)
            '''))
    
    st.plotly_chart(
        plot_reliability(df_wide_dict[domain], domain)
    )
    st.markdown(dedent('''
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    To evaluate whether performance changes depend on the participants' overall skill level without the confounding effects of mathematical coupling and regression to the mean, 
    **Oldham's method (1962)** was applied by regressing the difference between sessions against their mean. 
    A significant negative correlation indicates that lower-performing individuals achieved greater improvements (or reflects a ceiling effect among high performers), 
    whereas a non-significant correlation indicates that the magnitude of change is independent of baseline performance.
    - Oldham, P. D. (1962). A note on the analysis of repeated measurements of the same subjects. *Journal of Chronic Diseases, 15*, 969-977.
    '''))


def section_correlation(df_filt1: pd.DataFrame, df_filt2: pd.DataFrame):
    st.markdown(
        "### Correlations between scores " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":yellow-badge[:material/change_history: session filtered] "
    )
    c1, c2, c3, c4 = st.columns([2, 3, 2, 4])
    with c1:
        scores = st.selectbox("Scores", ["domain", "raw"])
    with c2:
        level = st.selectbox("Level", ["Between-person (S1)", "Between-person (mean)", "Within-person"])
        is_within = (level == "Within-person")
    with c3:
        method = st.selectbox("Method", ["pearson", "spearman"], disabled=is_within)
    with c4:
        controls = st.multiselect("Control for", ["Age", "IsFemale", "EduYears"], disabled=is_within)

    c1, c2, c3 = st.columns([3, 2, 5])
    with c1:
        with_moca = st.checkbox("Also correlate with MoCA", value=True)
    with c2:
        by_age_group = st.checkbox("One per age group", value=False)

    if by_age_group and (scores == "domain"):
        with c3:
            show_pc1 = st.checkbox(
                "Show PC1 of the correlations between domain scores", value=True
                # value=False, disabled=(not by_age_group) or (scores != "domain")
            )
    else:
        show_pc1 = False

    if is_within:
        df = df_filt1
    elif level == "Between-person (mean)":
        df = select_session(df_filt1, "mean")
    else:  # Between-person (S1)
        df = df_filt1.query("Session == 1")

    cols = P_DOMAINS if scores == "domain" else P_RAW_SCORES
    cols = ["MoCA"] + cols if with_moca else cols

    args = [cols, method, controls, is_within]
    corr_dfs = {}
    if by_age_group:
        for g, grp_df in df.groupby("AgeGroup", observed=True):
            corr_dfs[g] = calc_correlations(grp_df, *args)
    else:
        corr_dfs["Overall"] = calc_correlations(df, *args)

    fig = plot_corr_heatmap(corr_dfs, cols)
    if show_pc1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig, use_container_width=False)
        with c2:
            st.markdown(
                "#### PC1 of the correlations between domain scores "
            )
            df_pc1 = calc_corr_pc1(corr_dfs, P_DOMAINS)
            st.plotly_chart(
                plot_pc1_bars(df_pc1), 
                use_container_width=False
            )
    else:
        st.plotly_chart(fig, use_container_width=False)


def section_individual(df_filt2: pd.DataFrame):
    st.markdown(
        "### Individual profile " + 
        ":blue-badge[:material/check: age filtered] " + 
        ":blue-badge[:material/check: session filtered] "
    )
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        k = st.slider("Number of clusters", min_value=2, max_value=6, step=1)
    with c2:
        clust_method = st.selectbox("Clustering method", ["K-Means", "Gaussian mixture"])
    with c3:
        proj_method = st.selectbox("Projection method", ["PCA", "t-SNE"])
    with c4:
        sid = st.selectbox("Participant", df_filt2["SID"].unique().tolist())

    scaler = StandardScaler().fit(df_filt2[P_DOMAINS]) 

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Clustering + 2-D projection")
        df_2d, df_centers = run_clust_and_proj(df_filt2, scaler, P_DOMAINS, clust_method, proj_method, k)
        if df_2d is not None:
            st.plotly_chart(
                plot_clusters(
                    df_2d, df_centers, proj_method, 
                    sid if df_filt2.query(f"SID == '{sid}'")[P_DOMAINS].notna().all(axis=None) else None
                )
            )
    with c2:
        st.markdown("#### Z-transformed domain scores")
        st.plotly_chart(
            plot_radar(df_filt2, scaler, sid, P_DOMAINS)
        )


## Main ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Platform Data Summary", layout="wide")
    st.title("Platform Data Summary")

    df_filt2, df_filt1 = setup_data_sidebar()

    tabs = st.tabs([
        "Overview", "Cross-sectional", "Longitudinal", 
        "Reliability", "Correlation", "Individual"
    ])
    with tabs[0]:
        section_overview(df_filt1, df_filt2)
    with tabs[1]:
        section_cross_sectional(df_filt2)
    with tabs[2]:
        section_longitudinal(df_filt1)
    with tabs[3]:
        section_reliability(df_filt1)
    with tabs[4]:
        section_correlation(df_filt1, df_filt2)
    with tabs[5]:
        section_individual(df_filt2)


if __name__ == "__main__":
    # st.html(
    #     """
    #     <style>
    #     [data-testid="stCaptionContainer"] {
    #         line-height: 1.1 !important; 
    #         margin-top: 0px !important; 
    #         margin-bottom: 0px !important; 
    #     }
    #     </style>
    #     """
    # )
    main()
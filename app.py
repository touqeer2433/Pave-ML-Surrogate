"""Streamlit web app: flexible pavement critical strains and fatigue/rutting life.

Run locally:  streamlit run app.py
"""
import io

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict import Predictor

matplotlib.use("Agg")
st.set_page_config(page_title="Pavement Strain & Life Predictor", page_icon="🛣️", layout="wide")


@st.cache_resource
def load():
    return Predictor()


P = load()
D, R = P.defaults(), P.ranges()

LAYERS = [  # key, name, colour
    ("WC", "Wearing course (AC)", "#3a3a3a"),
    ("ACB", "Asphalt base (AC)", "#5c5c5c"),
    ("AggB", "Aggregate base", "#c9a66b"),
    ("SB", "Subbase", "#d8bf8f"),
    ("Fill", "Subgrade fill", "#a88b6a"),
    ("SG", "Natural subgrade", "#8c6f52"),
]
STRAIN = {"WC": "et_WC", "ACB": "et_ACB", "AggB": "ev_AggB", "SB": "ev_SB", "Fill": "ev_Fill", "SG": "ev_SG"}
UNBOUND = ["AggB", "SB", "Fill", "SG"]
LABEL = {"φ": "Friction angle φ (°)", "K": "Flow-stress ratio K", "ψ": "Dilation angle ψ (°)", "σy": "Cohesion / yield stress σy (MPa)"}


def num(key, label, fmt="%.3f"):
    lo, hi = R[key]
    return st.number_input(label, value=float(round(D[key], 4)), min_value=float(lo) * 0.5,
                           max_value=float(hi) * 1.5, format=fmt, key=key,
                           help=f"Database range: {lo:.4g} – {hi:.4g}")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Design inputs")
x = {}
with st.sidebar.expander("Loading and climate", expanded=True):
    x["P"] = num("P", "Tyre pressure P (MPa)")
    x["T"] = num("T", "Temperature T (°C)", "%.1f")
    reps = st.number_input("Design load repetitions", value=1.0e6, min_value=1.0, format="%.3e")
for k, name, _ in LAYERS:
    with st.sidebar.expander(name, expanded=k in ("WC", "ACB")):
        x[f"t_{k}"] = num(f"t_{k}", "Thickness (mm)", "%.1f")
        x[f"E_{k}"] = num(f"E_{k}", "Elastic modulus E (MPa)", "%.1f")
        x[f"ν_{k}"] = num(f"ν_{k}", "Poisson's ratio ν", "%.3f")
        if k in UNBOUND:
            for p in ("φ", "K", "ψ", "σy"):
                x[f"{p}_{k}"] = num(f"{p}_{k}", LABEL[p], "%.4f" if p == "σy" else "%.2f")

# ---------------------------------------------------------------- header
st.title("Flexible Pavement Strain & Life Predictor")
st.caption("ANN-ensemble surrogate of 3,500 three-dimensional finite element analyses. "
           "Strains in microstrain (+ tension, − compression). Lives from Asphalt Institute transfer functions.")

tab1, tab2, tab3 = st.tabs(["Single design", "Batch prediction (CSV)", "About"])

# ---------------------------------------------------------------- single design
with tab1:
    r = P.predict(x, reps).iloc[0]
    if r["out_of_range_inputs"]:
        st.warning(f"Outside the database range (prediction not reliable): {r['out_of_range_inputs']}")

    c = st.columns(4)
    c[0].metric("Fatigue life Nf", f"{r['Nf']:.3g}")
    c[1].metric("Rutting life Nr (fill)", f"{r['Nr_fill']:.3g}")
    c[2].metric("Rutting life Nr (subgrade)", f"{r['Nr_subgrade']:.3g}")
    c[3].metric("Damage ratio", f"{r['damage_ratio']:.3f}")
    if r["adequate"]:
        st.success(f"Section ADEQUATE for {reps:.3g} repetitions. Governing mode: {r['governing_mode']}.")
    else:
        st.error(f"Section NOT adequate for {reps:.3g} repetitions. Governing mode: {r['governing_mode']}.")

    left, right = st.columns([1.15, 1])
    with left:  # cross-section drawn to scale
        th = [x[f"t_{k}"] for k, _, _ in LAYERS]
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        y, H = 0.0, sum(th)
        last = 0.05 * H
        for (k, name, col), t in zip(LAYERS, th):
            ax.add_patch(plt.Rectangle((0, y - t), 1, t, fc=col, ec="k", lw=0.8))
            ly = min(y - t / 2, last - 0.05 * H)  # keep labels of thin layers apart
            last = ly
            ax.text(-0.03, ly, f"{name}, {t:.0f} mm", ha="right", va="center", fontsize=8.5)
            e = r[STRAIN[k] + "_microstrain"]
            yy = y - t if k in ("WC", "ACB") else y  # bottom of AC, top of unbound
            ax.plot(0.5, yy, "o", ms=7, mfc="red", mec="white", zorder=5)
            ax.annotate(f"{'εt' if k in ('WC', 'ACB') else 'εv'} = {e:.1f} µε", (0.5, yy), (1.05, yy + (0.025 if k in ("WC", "ACB") else -0.025) * H),
                        va="center", fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="red", lw=0.8))
            y -= t
        ax.annotate("", (0.5, 0), (0.5, H * 0.08), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="navy"))
        ax.text(0.5, H * 0.09, f"P = {x['P']:.2f} MPa", ha="center", va="bottom", fontsize=9, color="navy")
        ax.set_xlim(-0.75, 1.75); ax.set_ylim(-H * 1.02, H * 0.16); ax.axis("off")
        ax.set_title("Cross-section (to scale) and critical strains", fontsize=10)
        st.pyplot(fig, width="stretch")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("Download figure (PNG)", buf.getvalue(), "cross_section.png", "image/png")
    with right:
        tbl = pd.DataFrame({
            "Response": ["εt bottom of wearing course", "εt bottom of asphalt base", "εv top of aggregate base",
                         "εv top of subbase", "εv top of subgrade fill", "εv top of natural subgrade",
                         "Fatigue life Nf", "Fatigue life Nf (direct ANN)", "Rutting life Nr, fill",
                         "Rutting life Nr, subgrade", "Allowable repetitions", "Damage ratio"],
            "Value": [f"{r[s + '_microstrain']:.1f} µε" for s in P.meta["strains"]]
                     + [f"{r[k]:.3g}" for k in ("Nf", "Nf_direct", "Nr_fill", "Nr_subgrade", "N_allowable", "damage_ratio")],
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption(f"Ensemble spread (log10): εt {r['ensemble_spread_log10_et']:.4f}, "
                   f"εv subgrade {r['ensemble_spread_log10_ev_subgrade']:.4f}")

# ---------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with one design per row. Column names must match the model inputs "
             "(e.g. `t_ACB`, `E_ACB`, `P`). Missing columns take the database median.")
    tmpl = pd.DataFrame([D]).round(4)
    st.download_button("Download input template", tmpl.to_csv(index=False).encode("utf-8-sig"),
                       "input_template.csv", "text/csv")
    reps_b = st.number_input("Design repetitions (batch)", value=1.0e6, min_value=1.0, format="%.3e", key="rb")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        dfin = pd.read_csv(up, encoding="utf-8-sig")
        res = pd.concat([dfin.reset_index(drop=True), P.predict(dfin, reps_b)], axis=1)
        st.dataframe(res, width="stretch")
        st.download_button("Download results", res.to_csv(index=False).encode("utf-8-sig"), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab3:
    st.markdown(f"""
**Model.** Ensemble of {P.meta['n_members']} feed-forward networks (4 × 256, SiLU) trained on 3,465 three-dimensional
FE analyses of six-layer flexible pavements (43 inputs).

**Test accuracy (R²).** εt WC {P.meta['test_R2']['et_WC']:.3f} · εt ACB {P.meta['test_R2']['et_ACB']:.3f} ·
εv AggB {P.meta['test_R2']['ev_AggB']:.3f} · εv SB {P.meta['test_R2']['ev_SB']:.3f} ·
εv Fill {P.meta['test_R2']['ev_Fill']:.3f} · εv SG {P.meta['test_R2']['ev_SG']:.3f}

**Transfer functions (Asphalt Institute).**
Nf = 0.0796 · εt^−3.291 · E_ACB(psi)^−0.854  Nr = 1.365×10⁻⁹ · εv^−4.477

**Validity.** Inputs must lie inside the database ranges (hover over any input to see its range).
Density and saturation do not affect the FE responses and are held at their median values.

**Citation.** Rind, T. A., et al. Machine learning surrogates of three-dimensional finite element analysis for
layer-wise critical strains and fatigue and rutting life of flexible pavements (submitted).
""")

"""Streamlit front-end for the smell-agent odor predictor.

Run from the project root:

    source .venv/bin/activate
    streamlit run app.py

The heavy lifting lives in predict_smiles.py; this file only handles layout,
input validation and presentation.
"""

import base64
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

import predict_smiles as ps

st.set_page_config(
    page_title="smell-agent",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLES = {
    "Ethanol": "CCO",
    "Banana": "CC(C)CCOC(C)=O",
    "Vanilla": "COc1cc(C=O)ccc1O",
    "Mint": "CC(C)C1CCC(C)CC1O",
    "Almond": "O=Cc1ccccc1",
    "Lemon": "CC(=CCC1)C1=C(C)C",
}

st.markdown(
    """
<style>
  :root {
    --accent: #2F6FED;
    --ink: #111827;
    --muted: #6B7280;
    --line: #E5E7EB;
  }

  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
                 "Helvetica Neue", Arial, sans-serif;
  }

  .block-container {
    max-width: 1040px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

  /* Hero */
  .hero { margin-bottom: 1.4rem; }
  .hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--accent);
    background: #EAF1FE;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    margin-bottom: 0.7rem;
  }
  .hero h1 {
    font-size: 2.15rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    color: var(--ink);
  }
  .hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
  }

  /* Cards */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #FFFFFF;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    padding: 0.35rem 0.4rem;
  }

  /* Inputs */
  .stTextArea textarea {
    border-radius: 10px !important;
    border-color: var(--line) !important;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.95rem;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(47, 111, 237, 0.12) !important;
  }
  .stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid var(--line);
    transition: all 0.15s ease;
  }
  .stButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .stButton > button[kind="primary"]:hover {
    background: #1F5BD6;
    border-color: #1F5BD6;
  }

  /* Rank cards */
  .rank-card {
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: linear-gradient(180deg, #FBFCFE 0%, #FFFFFF 100%);
    height: 100%;
  }
  .rank-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }
  .rank-name {
    font-size: 1.15rem;
    font-weight: 650;
    color: var(--ink);
    margin-top: 0.2rem;
    text-transform: capitalize;
  }
  .rank-score {
    font-size: 0.9rem;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
    margin-top: 0.1rem;
  }

  .muted { color: var(--muted); font-size: 0.86rem; }
  .structure {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #FFFFFF;
    padding: 0.4rem;
    text-align: center;
  }
  .hint {
    border-left: 3px solid var(--accent);
    background: #F8FAFF;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    color: #374151;
    font-size: 0.9rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model loading
@st.cache_resource(show_spinner=False, max_entries=2)
def get_models(device: str):
    return ps.load_models(device)


def mol_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(360, 260))


# ------------------------------------------------------------------- session
st.session_state.setdefault("result", None)

# --------------------------------------------------------------------- header
st.markdown(
    """
<div class="hero">
  <span class="hero-badge">MPNN-POM · 10-model ensemble</span>
  <h1>What does a molecule smell like?</h1>
  <p>Enter a SMILES string to get its odor profile across 138 smell words.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")

    import torch

    has_gpu = torch.cuda.is_available()
    options = (["GPU (CUDA)", "CPU"] if has_gpu else ["CPU"])
    device_choice = st.radio("Device", options, index=0, key="device")
    device = "cuda" if device_choice.startswith("GPU") else "cpu"

    top_n = st.slider("Smells to show", min_value=5, max_value=30, value=10, step=1)

    st.divider()
    st.markdown("### Model")
    st.caption(
        "5 message-passing rounds, 256-dim odor map, "
        "trained on the OpenPOM curated set (4,983 molecules)."
    )
    st.caption("Single model test ROC-AUC: **0.875**")
    if has_gpu:
        st.caption("GPU: " + torch.cuda.get_device_name(0))
    else:
        st.caption("Running on CPU — first prediction takes ~20 s.")

    st.divider()
    with st.expander("How it works"):
        st.markdown(
            "Each SMILES is turned into an atom graph, the MPNN embeds it, and "
            "the readout predicts a probability per smell word. Ten independently "
            "trained models are averaged for the final profile."
        )

# ---------------------------------------------------------------------- input
left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### Molecule")
        smiles = st.text_area(
            "SMILES",
            key="smiles",
            height=96,
            placeholder="e.g. CC(C)CCOC(C)=O",
            label_visibility="collapsed",
        )

        st.markdown('<div class="muted">Try an example</div>', unsafe_allow_html=True)
        example_names = list(EXAMPLES)
        for start in range(0, len(example_names), 3):
            cols = st.columns(3)
            for col, name in zip(cols, example_names[start:start + 3]):
                with col:
                    st.button(
                        name,
                        key=f"example_{name}",
                        width="stretch",
                        on_click=lambda s=EXAMPLES[name]: st.session_state.update(smiles=s),
                    )

        predict_clicked = st.button(
            "Predict smell", type="primary", width="stretch", key="predict_btn"
        )

        st.markdown('<div class="muted" style="margin-top:.6rem">Structure</div>',
                    unsafe_allow_html=True)
        preview = smiles.strip() if smiles else ""
        img = mol_image(preview) if preview else None
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(
                f'<div class="structure"><img src="data:image/png;base64,{b64}" '
                f'style="width:100%;max-width:360px"/></div>',
                unsafe_allow_html=True,
            )
        elif preview:
            st.warning("Not a valid SMILES string.")
        else:
            st.markdown(
                '<div class="structure" style="padding:2.2rem;color:#9CA3AF">'
                "Structure preview</div>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------- predict
if predict_clicked:
    value = smiles.strip()
    if not value:
        st.error("Enter a SMILES string first.")
    elif "," in value:
        st.error("One molecule at a time, please (no commas).")
    elif Chem.MolFromSmiles(value) is None:
        st.error("That SMILES string could not be parsed.")
    else:
        try:
            with st.spinner("Loading the ensemble and predicting…"):
                models = get_models(device)
                scores = np.asarray(ps.predict(models, value)).squeeze()
        except Exception as exc:  # surface load/predict problems in the UI
            st.exception(exc)
        else:
            st.session_state.result = {
                "smiles": value,
                "scores": [float(x) for x in scores],
            }

# -------------------------------------------------------------------- results
with right:
    with st.container(border=True):
        st.markdown("#### Odor profile")
        result = st.session_state.result

        if result is None:
            st.markdown(
                '<div class="hint">Run a prediction to see the ranked odor words '
                "here.</div>",
                unsafe_allow_html=True,
            )
        else:
            frame = pd.DataFrame(
                {"odor": ps.TASKS, "score": result["scores"]}
            ).sort_values("score", ascending=False).reset_index(drop=True)

            top3 = frame.head(3)
            cols = st.columns(3)
            for rank, (col, (_, row)) in enumerate(zip(cols, top3.iterrows()), start=1):
                with col:
                    st.markdown(
                        f'<div class="rank-card">'
                        f'<div class="rank-label">Rank {rank}</div>'
                        f'<div class="rank-name">{row.odor}</div>'
                        f'<div class="rank-score">{row.score:.3f}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            shown = frame.head(top_n)
            base = alt.Chart(shown).encode(
                y=alt.Y("odor:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=160, labelFontSize=12)),
                x=alt.X("score:Q", title=None,
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".1f", grid=True, tickCount=6)),
                tooltip=[
                    alt.Tooltip("odor:N", title="smell"),
                    alt.Tooltip("score:Q", title="score", format=".3f"),
                ],
            )
            bars = base.mark_bar(color="#2F6FED", cornerRadiusEnd=4, height=14)
            labels = base.mark_text(
                align="left", dx=4, color="#374151", fontSize=11
            ).encode(text=alt.Text("score:Q", format=".3f"))
            chart = (bars + labels).properties(
                height=max(300, 26 * len(shown)),
                padding={"left": 5, "right": 20, "top": 5, "bottom": 5},
            )
            st.altair_chart(chart, width="stretch")

            st.caption(f"SMILES: `{result['smiles']}`")
            with st.expander(f"All {len(frame)} odor scores"):
                st.dataframe(
                    frame.assign(score=lambda d: d.score.round(4)),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "odor": "Smell",
                        "score": st.column_config.NumberColumn("Score", format="%.4f"),
                    },
                )

            stem = "".join(c for c in result["smiles"] if c.isalnum())[:20] or "molecule"
            csv_bytes = frame.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(
                [
                    {"odor": r.odor, "score": round(float(r.score), 5)}
                    for r in frame.itertuples()
                ],
                indent=2,
            ).encode("utf-8")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=f"odors_{stem}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with dl2:
                st.download_button(
                    "Download JSON",
                    data=json_bytes,
                    file_name=f"odors_{stem}.json",
                    mime="application/json",
                    width="stretch",
                )

st.markdown(
    '<div class="muted" style="margin-top:1.4rem;text-align:center">'
    "smell-agent · MPNN-POM odor prediction · scores are per-word probabilities, "
    "not concentrations"
    "</div>",
    unsafe_allow_html=True,
)

# app.py
# pip install dash plotly pandas numpy scikit-learn scipy

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import re
from collections import Counter

# Expected 5 countries per continent (based on dataset design)
EXPECTED_COUNTRIES = {
    "Africa": ["Egypt", "Madagascar", "Morocco", "Seychelles", "South Africa"],
    "Asia": ["Israel", "Japan", "Korea, Republic of", "Singapore", "United Arab Emirates"],
    "Europe": ["France", "Germany", "Netherlands", "Switzerland", "United Kingdom"],
    "North America": ["Bahamas", "Canada", "Costa Rica", "Mexico", "United States"],
    "South America": ["Argentina", "Brazil", "Chile", "Colombia", "Uruguay"],
    "Oceania": ["Australia", "Fiji", "New Zealand", "Samoa", "Tonga"],
}

# --- Alias map for country name normalization ---
ALIASES = {
    "korea republic of": "korea republic of",  # anchor
    "republic of korea": "korea republic of",
    "south korea": "korea republic of",
    "uae": "united arab emirates",
    "u a e": "united arab emirates",
    "united states of america": "united states",
    "usa": "united states",
    "u s a": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "the bahamas": "bahamas",
}

def _norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)

# ---------- Helpers ----------
def autodetect_columns(df):
    # Prefer Country / Countries for labels
    label_col = None
    for c in df.columns:
        if c.strip().lower() in ["country", "countries"]:
            label_col = c
            break
    if label_col is None:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        label_col = non_numeric[0] if non_numeric else df.columns[0]
    # Continent / Continents
    cont_col = None
    for c in df.columns:
        if c.strip().lower() in ["continent", "continents"]:
            cont_col = c
            break
    return label_col, cont_col

def clean_numeric_columns(df: pd.DataFrame, exclude_cols):
    """Coerce numeric-looking columns to floats by stripping commas, %, and N/A text.
    Returns the list of numeric feature columns after cleaning.
    """
    def looks_numeric(s: pd.Series) -> bool:
        vals = s.astype(str)
        non_empty = vals.str.strip() != ""
        if non_empty.sum() == 0:
            return False
        sample = vals[non_empty].head(50)
        pat = re.compile(r"^[\s\-\+\.,\d%]+$")
        return sample.apply(lambda x: bool(pat.match(x))).mean() >= 0.6

    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype.kind in "biufc" or looks_numeric(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("N/A", "", regex=False)
                .str.replace("NA", "", regex=False)
                .str.replace("nan", "", regex=False)
                .str.strip()
                .replace({"": np.nan})
                .infer_objects(copy=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Final feature list = all numeric columns except excluded ones
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    return feature_cols

def load_and_clean(file_path: str):
    if not os.path.exists(file_path):
        return None, None, None, None
    df = pd.read_csv(file_path)
    original_len = len(df)
    label_col, cont_col = autodetect_columns(df)
    exclude = set([label_col] if label_col else []) | set([cont_col] if cont_col else []) | {"year", "Year"}
    feature_cols = clean_numeric_columns(df, exclude)
    # Normalize labels/continents to avoid 'nan' text in hover
    if label_col:
        df[label_col] = (
            df[label_col]
              .astype("string")
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .fillna("Unknown")
              .replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown", "None": "Unknown"})
        )
    if cont_col:
        df[cont_col] = (
            df[cont_col]
              .astype("string")
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )
        # Forward-fill continent headers for rows where the CSV leaves continent blank under a header
        df[cont_col] = df[cont_col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA}).ffill()
    # --- Strict row filtering: keep only valid country rows ---
    if label_col and cont_col:
        # drop rows with missing label/continent
        df = df[df[label_col].notna() & (df[label_col] != "") & df[cont_col].notna() & (df[cont_col] != "")].copy()
        # allow only the six known continents
        allowed_continents = {"Africa", "Asia", "Europe", "North America", "South America", "Oceania"}
        df = df[df[cont_col].isin(allowed_continents)].copy()
        # de-duplicate countries just in case
        df = df.drop_duplicates(subset=[label_col], keep="first")
        # Build the exact 30-country set using expected names per continent with tolerant matching
        df["__norm_country__"] = df[label_col].apply(_norm_name)
        selected_rows = []
        missing_report = []
        for cont, expected_list in EXPECTED_COUNTRIES.items():
            sub = df[df[cont_col] == cont]
            for exp_name in expected_list:
                target = _norm_name(exp_name)
                # exact normalized match
                hit = sub[sub["__norm_country__"] == target]
                if hit.empty:
                    # try alias-normalized match based on raw label
                    sub_alias = sub.copy()
                    sub_alias["__norm_country__"] = sub_alias[label_col].apply(_norm_name)
                    hit = sub_alias[sub_alias["__norm_country__"] == target]
                if hit.empty:
                    # startswith on normalized
                    hit = sub_alias[sub_alias["__norm_country__"].str.startswith(target.split(" ")[0], na=False)]
                if hit.empty and len(target) >= 5:
                    # contains on a 5-char stem
                    stem = target[:5]
                    hit = sub_alias[sub_alias["__norm_country__"].str.contains(re.escape(stem), na=False)]
                if hit.empty:
                    missing_report.append((cont, exp_name))
                else:
                    selected_rows.append(hit.iloc[0])
        if selected_rows:
            df = pd.DataFrame(selected_rows).reset_index(drop=True)
        df.drop(columns=["__norm_country__"], errors="ignore", inplace=True)
        if missing_report:
            print(f"⚠ Missing expected countries in {file_path}: {missing_report}")
    feature_cols = clean_numeric_columns(df, exclude)

    # Enforce exactly 5 per continent when possible (expected 6 x 5 = 30)
    # (removed: now we select the expected rows above)
    if cont_col in df.columns:
        df[cont_col] = df[cont_col].fillna("Unknown")

    if cont_col in df.columns:
        counts = df[cont_col].value_counts().to_dict()
        print(f"Loaded {file_path}: total={len(df)} per-continent={counts}")
    return df, label_col, cont_col, feature_cols

def audit_dataframe(year, path, df, label_col, cont_col, feature_cols):
    print(f"\n=== AUDIT {year} ===")
    print(f"File: {path}")
    if df is None:
        print("Status: MISSING FILE"); return
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Label column: {label_col}")
    print(f"Continent column: {cont_col}")
    if cont_col:
        vc = df[cont_col].value_counts(dropna=False)
        print("Continents value counts:\n", vc.to_string())
    print(f"Detected numeric feature count (after cleaning): {len(feature_cols)}")
    print("First 5 features:", feature_cols[:5])
    blank_labels = df[label_col].isna().sum() if label_col else 'n/a'
    blank_conts = df[cont_col].isna().sum() if cont_col else 'n/a'
    print(f"Blank labels: {blank_labels} | Blank continents: {blank_conts}")

def knn_graph(X, k=10, metric="euclidean", symmetrize="min",
              weight_scheme="heat", sigma=None, local_scale=True):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric=metric).fit(X)
    dists, idxs = nn.kneighbors(X)
    if idxs.shape[1] > 0 and np.all(idxs[:,0] == np.arange(n)):
        dists, idxs = dists[:,1:], idxs[:,1:]
    rows = np.repeat(np.arange(n)[:,None], idxs.shape[1], axis=1).ravel()
    cols = idxs.ravel()

    if weight_scheme == "binary":
        vals = np.ones(rows.size, dtype=float)
    else:
        if local_scale:
            sigma_i = np.maximum(dists[:,-1], 1e-12)
            vals = np.exp(-(dists**2) / np.maximum(sigma_i[:,None] * sigma_i[idxs], 1e-12))
        else:
            if sigma is None:
                sigma = float(np.median(dists)) if np.isfinite(np.median(dists)) else 1.0
                if sigma <= 0: sigma = 1.0
            vals = np.exp(-(dists**2)/(sigma**2 + 1e-12))
        vals = vals.ravel()

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T) if symmetrize == "max" else W.minimum(W.T)
    W.setdiag(0.0); W.eliminate_zeros()
    return W

def laplacian(W, normalized=True):
    d = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        d_inv_sqrt = np.power(d, -0.5, where=d>0)
        D_inv_sqrt = diags(d_inv_sqrt)
        I = diags(np.ones(W.shape[0]))
        return I - (D_inv_sqrt @ W @ D_inv_sqrt)
    else:
        return diags(d) - W

def laplacian_eigenmap(L, n_components=3):
    k = min(n_components + 1, L.shape[0] - 1)
    evals, evecs = eigsh(L, k=k, which="SM")
    order = np.argsort(evals)
    evecs = evecs[:, order][:, 1:n_components+1]  # drop trivial vector
    return evecs

def compute_embedding(df: pd.DataFrame, label_col: str, cont_col: str, feature_cols: list, k_graph=10, n_components=3, k_anchor=5):
    if df is None or label_col is None or cont_col is None or not feature_cols:
        return pd.DataFrame(columns=["Country","Continent","LE1","LE2","LE3","is_stray"]) 
    if len(df) == 0:
        return pd.DataFrame(columns=["Country","Continent","LE1","LE2","LE3","is_stray"]) 

    # Drop features that are entirely NaN for this year
    valid_feats = [c for c in feature_cols if not df[c].isna().all()]
    if len(valid_feats) == 0:
        return pd.DataFrame(columns=["Country","Continent","LE1","LE2","LE3","is_stray"]) 

    labels = df[label_col].astype(str).fillna("Unknown").values
    continents = df[cont_col].astype(str).fillna("Unknown").values

    X = df[valid_feats].to_numpy()
    # Impute & scale
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    n = X.shape[0]
    k_graph = max(2, min(k_graph, n-1))

    # Graph + largest CC
    W = knn_graph(X, k=k_graph, symmetrize="min", weight_scheme="heat", local_scale=True)
    _, cc_labels = connected_components(W, directed=False)
    largest = np.argmax(np.bincount(cc_labels))
    keep_idx = np.where(cc_labels == largest)[0]
    drop_idx = np.setdiff1d(np.arange(n), keep_idx)

    # Eigenmap on largest CC
    W_cc = W[keep_idx][:, keep_idx]
    L_cc = laplacian(W_cc, normalized=True)
    Y_cc = laplacian_eigenmap(L_cc, n_components=n_components)

    # Fill + anchor strays via heat kernel
    emb = np.full((n, n_components), np.nan)
    emb[keep_idx] = Y_cc

    # --- Global rescale to fit [-0.3, 0.3] cube without changing relative geometry ---
    max_abs = np.nanmax(np.abs(emb))
    if np.isfinite(max_abs) and max_abs > 0:
        scale = 0.28 / max_abs  # leave some margin inside the 0.3 bound
        emb *= scale

    if len(drop_idx):
        k_anchor = min(k_anchor, len(keep_idx))
        nbrs_main = NearestNeighbors(n_neighbors=k_anchor).fit(X[keep_idx])
        for i in drop_idx:
            dists, idxs_local = nbrs_main.kneighbors(X[i].reshape(1,-1), return_distance=True)
            dists = dists.ravel(); idxs_local = idxs_local.ravel()
            med = np.median(dists)
            sigma = float(med) if np.isfinite(med) and med > 1e-12 else 1.0
            w = np.exp(-(dists**2) / (sigma**2))
            if not np.isfinite(w).all() or w.sum() == 0:
                w = np.ones_like(w)
            w = w / w.sum()
            abs_idx = keep_idx[idxs_local]
            neigh = emb[abs_idx]
            mask = ~np.isnan(neigh).any(axis=1)
            if mask.sum() == 0:
                emb[i] = 0.0
            else:
                emb[i] = np.average(neigh[mask], axis=0, weights=w[mask])

    # Final guard: no NaNs left
    nan_rows = np.isnan(emb).any(axis=1)
    if nan_rows.any():
        emb[nan_rows] = 0.0

    out = pd.DataFrame({
        "Country": labels,
        "Continent": continents,
        "LE1": emb[:,0],
        "LE2": emb[:,1],
        "LE3": emb[:,2],
        "is_stray": False
    })
    out.loc[drop_idx, "is_stray"] = True
    return out

# ---------- Precompute ----------
BASE = "/Users/evankimeme/Desktop/Ardmore/[Ardmore] Laplacian Eigenmap"
YEARS = list(range(2018, 2025))

# Load & clean all years first (with audit)
dfs = {}
labels_cols = {}
cont_cols = {}
feature_sets = []
for y in YEARS:
    path = os.path.join(BASE, f"Laplacian Macro Analysis Data {y}.csv")
    df, lbl, cont, feats = load_and_clean(path)
    dfs[y] = df; labels_cols[y] = lbl; cont_cols[y] = cont
    if df is not None and lbl is not None and cont is not None and feats:
        feature_sets.append(set(feats))
    audit_dataframe(y, path, df, lbl, cont, feats or [])

# Common feature schema across years (intersection -> fallback to union)
COMMON_FEATURES = sorted(set.intersection(*feature_sets)) if feature_sets else []
if not COMMON_FEATURES and feature_sets:
    COMMON_FEATURES = sorted(set.union(*feature_sets))
print("\nCOMMON FEATURES (count):", len(COMMON_FEATURES))
print("Sample:", COMMON_FEATURES[:10])

# Compute embeddings using the fixed schema
embeddings = {}; sizes = {}
for y in YEARS:
    df = dfs[y]; lbl = labels_cols[y]; cont = cont_cols[y]
    emb = compute_embedding(df, lbl, cont, COMMON_FEATURES)
    embeddings[y] = emb
    sizes[y] = 0 if emb is None or emb.empty else len(emb)
    print(f"Year {y}: embedded rows = {sizes[y]}")

# Fixed colors for labels (not color strings as data)
COLOR_MAP = {
    "Africa": "red",
    "Asia": "orange",
    "Europe": "yellow",
    "North America": "green",
    "South America": "blue",
    "Oceania": "indigo",
    "Stray": "black",
    "Unknown": "black",
}

# ---------- Dash app ----------
app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth":"1100px","margin":"20px auto","fontFamily":"system-ui,-apple-system,Segoe UI,Roboto"},
    children=[
        html.H2("Laplacian Eigenmaps (3D) — 2018–2024", style={"marginBottom":"10px"}),
        html.Div([
            dcc.Dropdown(
                id="year-dd",
                options=[{"label": str(y), "value": y} for y in YEARS],
                value=2024, clearable=False, style={"width":"200px"}
            ),
        ], style={"margin":"10px 0"}),
        dcc.Graph(id="le-graph", style={"height":"700px"})
    ]
)

@app.callback(Output("le-graph","figure"), Input("year-dd","value"))
def update_graph(year):
    df = embeddings.get(year, pd.DataFrame())
    if df.empty:
        return px.scatter_3d(title=f"No data for {year}")

    # Group label for coloring: continents or "Stray"
    group = np.where(df["is_stray"], "Stray", df["Continent"])
    df = df.assign(Group=group)
    df["Group"] = df["Group"].fillna("Unknown").replace({"nan": "Unknown"})

    fig = px.scatter_3d(
        df, x="LE1", y="LE2", z="LE3",
        color="Group",
        color_discrete_map=COLOR_MAP,
        hover_name="Country",
        category_orders={"Group": ["Africa","Asia","Europe","North America","South America","Oceania","Stray","Unknown"]}
    )
    fig.for_each_trace(lambda t: t.update(marker=dict(color=COLOR_MAP.get(t.name, "black"))))
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
    # Title/status: how many points & breakdown
    total = len(df)
    unknown_cnt = int((df["Group"] == "Unknown").sum())
    stray_cnt = int((df["Group"] == "Stray").sum())
    counts = df["Group"].value_counts().to_dict()
    fig.update_layout(title=f"LE (3D) — {year}  ·  n={total}  ·  stray={stray_cnt}  ·  unknown={unknown_cnt}")
    fig.update_layout(
        legend_title_text="Continent / Status",
        scene=dict(
            xaxis=dict(title="LE1", range=[-0.3, 0.3], showspikes=False, autorange=False),
            yaxis=dict(title="LE2", range=[-0.3, 0.3], showspikes=False, autorange=False),
            zaxis=dict(title="LE3", range=[-0.3, 0.3], showspikes=False, autorange=False)
        ),
        scene_aspectmode='cube',
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='constant'  # completely freezes axis scaling
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)


"""
autooda: simple automatic EDA tool
Usage (CLI):
    python eda.py path/to/data.csv --outdir output --max_pairwise 6
Or import functions:
    from eda import generate_report
    generate_report("data.csv", outdir="output")
"""

import os
import sys
import argparse
import warnings
from datetime import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AutoEDA Report - {title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 30px; }}
    h1,h2,h3 {{ color: #2c3e50; }}
    .section {{ margin-bottom: 30px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background:#f4f6f8; text-align:left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin: 8px 0; }}
    .small {{ font-size: 0.9em; color: #666; }}
    .code {{ background:#f7f9fb; padding:6px; border-radius:4px; font-family:monospace; }}
  </style>
</head>
<body>
  <h1>AutoEDA Report - {title}</h1>
  <p class="small">Generated: {generated}</p>

  <div class="section">
    <h2>Dataset overview</h2>
    {overview_table}
  </div>

  <div class="section">
    <h2>Columns summary</h2>
    {columns_table}
  </div>

  <div class="section">
    <h2>Missing values</h2>
    {missing_table}
    <div><img src="{missing_img}" alt="missing-heatmap"></div>
  </div>

  <div class="section">
    <h2>Numeric column stats</h2>
    {numeric_table}
  </div>

  <div class="section">
    <h2>Categorical column top values</h2>
    {categorical_tables}
  </div>

  <div class="section">
    <h2>Correlation (Pearson)</h2>
    <div><img src="{corr_img}" alt="correlation-heatmap"></div>
    <p><strong>Top correlated pairs:</strong></p>
    {top_corr_table}
  </div>

  <div class="section">
    <h2>Distributions & outliers</h2>
    {dist_imgs}
  </div>

  <div class="section">
    <h2>Suggestions</h2>
    <ul>
      {suggestions}
    </ul>
  </div>

</body>
</html>
"""

# -------------------------
# Helper plotting functions
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_fig(fig, path, dpi=120):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def plot_missing_heatmap(df, outpath):
    fig = plt.figure(figsize=(10, max(2, df.shape[1]*0.25)))
    ax = fig.add_subplot(111)
    sns_matrix = df.isna().T.astype(int)  # shape: cols x rows
    im = ax.imshow(sns_matrix, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel("rows")
    ax.set_title("Missing values (black = missing)")
    save_fig(fig, outpath)

def plot_corr_heatmap(corr, outpath):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix (Pearson)")
    save_fig(fig, outpath)

def plot_numeric_distribution(series, outpath, bins=30):
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].hist(series.dropna(), bins=bins)
    axs[0].set_title(f"Histogram: {series.name}")
    axs[1].boxplot(series.dropna(), vert=False)
    axs[1].set_title(f"Boxplot: {series.name}")
    save_fig(fig, outpath)

def plot_categorical_bar(series, outpath, top_n=10):
    vc = series.value_counts().nlargest(top_n)
    fig = plt.figure(figsize=(6, max(3, len(vc)*0.3)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(vc)), vc.values)
    ax.set_yticks(range(len(vc)))
    ax.set_yticklabels(vc.index.astype(str))
    ax.invert_yaxis()
    ax.set_title(f"Top {len(vc)} categories for {series.name}")
    save_fig(fig, outpath)

# -------------------------
# EDA functions
# -------------------------
def dataset_overview(df):
    rows, cols = df.shape
    info = {
        "Rows": rows,
        "Columns": cols,
        "Memory (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        "Num missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
    }
    return info

def columns_summary(df):
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique(dropna=True))
        sample = df[col].dropna().head(3).astype(str).tolist()
        rows.append({
            "column": col,
            "dtype": dtype,
            "n_missing": n_missing,
            "n_unique": n_unique,
            "sample_values": ", ".join(sample)
        })
    return pd.DataFrame(rows)

def numeric_summary(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.DataFrame()
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurtosis()
    return desc

def categorical_top_values(df, top_n=8):
    cats = df.select_dtypes(include=['object', 'category', 'bool'])
    result = {}
    for c in cats.columns:
        result[c] = df[c].value_counts(dropna=False).head(top_n)
    return result

def top_correlations(df, n=10):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i,j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs_sorted[:n]

def detect_outliers_iqr(series, k=1.5):
    s = series.dropna()
    if s.empty or series.dtype == "object":
        return 0, None
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outliers = ((s < lower) | (s > upper)).sum()
    return int(outliers), (lower, upper)

# -------------------------
# HTML helpers
# -------------------------
def df_to_html_table(df, max_rows=20):
    if df is None or df.empty:
        return "<p>No data.</p>"
    return df.head(max_rows).to_html(classes="table", index=True, border=0)

def series_counts_to_html_table(series):
    if series is None or len(series) == 0:
        return "<p>No categorical columns.</p>"
    parts = []
    for name, vc in series.items():
        parts.append(f"<h3>{name}</h3>")
        parts.append(vc.to_frame().to_html(border=0))
    return "\n".join(parts)

# -------------------------
# Main report generator
# -------------------------
def generate_report(csv_path, outdir="output", title=None, max_pairwise=6):
    ensure_dir(outdir)
    images_dir = os.path.join(outdir, "images")
    ensure_dir(images_dir)

    df = pd.read_csv(csv_path)
    if title is None:
        title = os.path.basename(csv_path)

    # Overview
    overview_info = dataset_overview(df)
    overview_df = pd.DataFrame(list(overview_info.items()), columns=["metric","value"])

    # Columns summary
    cols_df = columns_summary(df)

    # Missing values
    missing_df = pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean().values * 100).round(2)
    }).sort_values("pct_missing", ascending=False)

    # Missing heatmap
    missing_img = "images/missing_heatmap.png"
    try:
        plot_missing_heatmap(df, os.path.join(outdir, missing_img))
    except Exception as e:
        # fallback: show a simple bar of missing percentages
        fig = plt.figure(figsize=(8, max(3, df.shape[1]*0.25)))
        ax = fig.add_subplot(111)
        ax.barh(range(len(missing_df)), missing_df["pct_missing"])
        ax.set_yticks(range(len(missing_df)))
        ax.set_yticklabels(missing_df["column"])
        ax.set_xlabel("% missing")
        save_fig(fig, os.path.join(outdir, missing_img))

    # Numeric stats
    num_stats = numeric_summary(df)

    # Categorical tops
    cat_tops = categorical_top_values(df)

    # Correlation
    num_df = df.select_dtypes(include=[np.number]).copy()
    corr_img = "images/correlation_heatmap.png"
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        plot_corr_heatmap(corr, os.path.join(outdir, corr_img))
        top_corr = top_correlations(df, n=10)
        top_corr_df = pd.DataFrame(top_corr, columns=["col1","col2","abs_pearson"]).round(3)
    else:
        # create an empty placeholder image
        fig = plt.figure(figsize=(6,2))
        fig.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        save_fig(fig, os.path.join(outdir, corr_img))
        top_corr_df = pd.DataFrame(columns=["col1","col2","abs_pearson"])

    # Distributions and outliers images
    dist_imgs_html = []
    # numeric columns: hist + boxplot
    for col in num_df.columns:
        imgname = f"images/dist_{col}.png"
        try:
            plot_numeric_distribution(num_df[col], os.path.join(outdir, imgname))
            outliers_count, bounds = detect_outliers_iqr(num_df[col])
            dist_imgs_html.append(f"<div><h4>{col} — outliers (IQR): {outliers_count}</h4><img src=\"{imgname}\"></div>")
        except Exception as e:
            continue

    # categorical bar charts
    for col, vc in cat_tops.items():
        imgname = f"images/cat_{col}.png"
        try:
            plot_categorical_bar(df[col].astype(str), os.path.join(outdir, imgname))
            dist_imgs_html.append(f"<div><h4>{col} — top categories</h4><img src=\"{imgname}\"></div>")
        except Exception:
            continue

    # pairwise scatter for up to max_pairwise numeric columns (choose highest variance)
    if num_df.shape[1] >= 2:
        var_sorted = num_df.var().sort_values(ascending=False)
        pair_cols = var_sorted.index[:max_pairwise].tolist()
        if len(pair_cols) >= 2:
            pair_img = "images/pairwise.png"
            fig = plt.figure(figsize=(4*len(pair_cols), 4*len(pair_cols)))
            # build scatter matrix manually to avoid pandas.plotting dependency
            from pandas.plotting import scatter_matrix
            _ = scatter_matrix(num_df[pair_cols].dropna(), alpha=0.5, diagonal='hist', figsize=(4*len(pair_cols), 4*len(pair_cols)))
            fig = plt.gcf()
            save_fig(fig, os.path.join(outdir, pair_img))
            dist_imgs_html.insert(0, f"<div><h4>Pairwise scatter (top variance cols)</h4><img src=\"{pair_img}\"></div>")

    # suggestions
    suggestions = []
    if missing_df["pct_missing"].max() > 50:
        suggestions.append("Some columns have >50% missing values — consider dropping or imputing them.")
    if overview_info["Duplicate rows"] > 0:
        suggestions.append("There are duplicate rows. Consider removing duplicates using df.drop_duplicates().")
    if num_df.shape[1] > 0:
        skewed = (num_df.skew().abs() > 1).sum()
        if skewed > 0:
            suggestions.append(f"{skewed} numeric columns are highly skewed (|skew| > 1). Consider log/sqrt transform.")
    if len(cat_tops) > 0:
        high_card = [c for c in cat_tops.keys() if df[c].nunique() > 100]
        if high_card:
            suggestions.append(f"Categorical columns with high cardinality: {', '.join(high_card)}. Consider encoding carefully.")

    if not suggestions:
        suggestions.append("No urgent issues detected. Review charts for deeper insight.")

    # Render HTML
    html = HTML_TEMPLATE.format(
        title=title,
        generated=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        overview_table=overview_df.to_html(index=False, border=0),
        columns_table=cols_df.to_html(index=False, border=0),
        missing_table=missing_df.to_html(index=False, border=0),
        missing_img=missing_img,
        numeric_table=num_stats.to_html(border=0) if not num_stats.empty else "<p>No numeric columns.</p>",
        categorical_tables=series_counts_to_html_table(cat_tops),
        corr_img=corr_img,
        top_corr_table=top_corr_df.to_html(index=False, border=0) if not top_corr_df.empty else "<p>No correlations available.</p>",
        dist_imgs="\n".join(dist_imgs_html),
        suggestions="\n".join(f"<li>{s}</li>" for s in suggestions)
    )

    out_html_path = os.path.join(outdir, "report.html")
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to: {out_html_path}")
    return out_html_path

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="AutoEDA - generate automatic EDA report for a CSV file")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--title", default=None, help="Title for report")
    parser.add_argument("--max_pairwise", type=int, default=6, help="Max numeric columns in pairwise scatter")
    args = parser.parse_args()
    generate_report(args.csv, outdir=args.outdir, title=args.title, max_pairwise=args.max_pairwise)

if __name__ == "__main__":
    main()

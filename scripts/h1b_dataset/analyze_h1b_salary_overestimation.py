from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROW_ID_COL = "ROW_ID"

ACTUAL_COL = "PREVAILING_WAGE"
PRED_COL = "estimated_salary_in_usd"
AI_BINARY_COL = "IS_AI"
AI_CONF_COL = "ai_confidence"

PARSE_STATUS_COL = "salary_parse_status"
RAW_OUTPUT_COL = "salary_raw_output"

BLOCK_COL = "BLOCK_ID"
JOB_KEY_COL = "JOB_CONTEXT_KEY"

DIFF_USD_COL = "predicted_minus_actual_usd"
PCT_POINTS_COL = "overestimation_pct_points"
RATIO_COL = "prediction_to_actual_ratio"
LOG_RATIO_COL = "log_prediction_minus_log_actual"

COMPLETED_PARSE_STATUSES = {
    "ok",
    "ok_legacy_missing_parse_status",
}

REQUIRED_PREDICTION_COLUMNS = {
    ROW_ID_COL,
    "JOB_TITLE",
    "SOC_CODE",
    "SOC_TITLE",
    "WORKSITE_STATE",
    "NAICS_CODE",
    "FULL_TIME_POSITION",
    "TOTAL_WORKER_POSITIONS",
    ACTUAL_COL,
    PRED_COL,
    AI_BINARY_COL,
}

REQUIRED_SAMPLED_INPUT_COLUMNS = {
    ROW_ID_COL,
    "JOB_TITLE",
    "SOC_CODE",
    "SOC_TITLE",
    "WORKSITE_STATE",
    "NAICS_CODE",
    "FULL_TIME_POSITION",
    "TOTAL_WORKER_POSITIONS",
    "PW_WAGE_LEVEL",
    ACTUAL_COL,
    AI_BINARY_COL,
}


@dataclass(frozen=True)
class AnalysisConfig:
    predictions_glob: str
    sampled_input: str
    scores_csv: str
    ai_high_threshold: float
    ai_low_threshold: float
    min_salary_usd: int
    max_salary_usd: int
    min_models: int
    primary_outcome: str
    primary_test: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple statistical analysis for H-1B salary overestimation: "
            "row-average percentage-point overestimation, Welch tests, and block-level contrasts."
        )
    )
    parser.add_argument(
        "--predictions-glob",
        default=(
            "data/h1b-lca-disclosure-data-2020-2024/"
            "sampled-h1b_fy2025_q4_qwen36_main_sampled_2000/*/"
            "llm_estimated_salaries-salary_estimator.csv"
        ),
        help="Glob for per-model salary-estimation CSV files.",
    )
    parser.add_argument(
        "--sampled-input",
        default="data/h1b_fy2025_q4_qwen36_main_sampled_2000.csv",
        help="Original sampled H-1B CSV used as estimator input.",
    )
    parser.add_argument(
        "--scores-csv",
        default="data/job_title_soc_qwen36_ai_confidence_scores.csv",
        help="Qwen3.6 job-context AI-confidence scores CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/h1b_salary_qwen36_main_analysis_simple",
        help="Directory for analysis outputs.",
    )
    parser.add_argument("--ai-high-threshold", type=float, default=0.80)
    parser.add_argument("--ai-low-threshold", type=float, default=0.30)
    parser.add_argument("--min-salary-usd", type=int, default=20_000)
    parser.add_argument("--max-salary-usd", type=int, default=1_000_000)
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument(
        "--fail-on-identical-prediction-vectors",
        action="store_true",
        help="Fail if two model outputs have identical prediction vectors.",
    )
    return parser.parse_args()


def import_stats():
    try:
        import scipy.stats as scipy_stats
    except ImportError as exc:
        raise RuntimeError("Install scipy first: uv add scipy") from exc
    return scipy_stats


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return re.sub(r"\s+", " ", s)


def stable_job_key(title: object, soc_code: object, soc_title: object) -> str:
    return f"{norm_text(title)} || {norm_text(soc_code)} || {norm_text(soc_title)}"


def clean_naics2(value: object) -> str:
    s = norm_text(value)
    digits = re.sub(r"\D", "", s)
    return digits[:2] if digits else ""


def make_default_block_id(df: pd.DataFrame) -> pd.Series:
    required = ["SOC_CODE", "WORKSITE_STATE", "NAICS_CODE", "FULL_TIME_POSITION", "PW_WAGE_LEVEL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot construct {BLOCK_COL}; missing columns: {missing}")

    return (
        df["SOC_CODE"].map(norm_text)
        + "||"
        + df["WORKSITE_STATE"].map(norm_text)
        + "||"
        + df["NAICS_CODE"].map(clean_naics2)
        + "||"
        + df["FULL_TIME_POSITION"].map(norm_text)
        + "||"
        + df["PW_WAGE_LEVEL"].map(norm_text)
    )


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)

    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "ai"}
    )


def parse_money_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return pd.to_numeric(s, errors="coerce")


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    return None


def slug_from_path(path: str | Path) -> str:
    p = Path(path)
    return p.parent.name if p.parent.name else p.stem


def model_family_from_slug(model_slug: str) -> str:
    s = str(model_slug).lower()

    if "qwen" in s:
        return "Qwen"
    if "llama" in s or "meta" in s:
        return "Llama"
    if "mistral" in s:
        return "Mistral"
    if "granite" in s or "ibm" in s:
        return "Granite"
    if "falcon" in s:
        return "Falcon"
    if "olmo" in s or "allenai" in s:
        return "Olmo"
    if "gemma" in s:
        return "Gemma"

    return "Other"


def safe_float(x: object) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def format_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    if parquet_path is not None:
        try:
            parquet_df = df.copy()

            # PyArrow is stricter than CSV: object columns cannot mix strings,
            # ints, floats, None, and pandas NA. For analysis output, object-like
            # columns are safest as nullable strings in Parquet.
            for col in parquet_df.columns:
                if (
                    parquet_df[col].dtype == "object"
                    or pd.api.types.is_string_dtype(parquet_df[col])
                ):
                    parquet_df[col] = parquet_df[col].astype("string")

            parquet_df.to_parquet(parquet_path, index=False)

        except ImportError:
            print(f"Skipping Parquet write because no Parquet engine is installed: {parquet_path}")

        except Exception as exc:
            print(
                f"Skipping Parquet write for {parquet_path} because of "
                f"{type(exc).__name__}: {exc}"
            )

def read_prediction_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required prediction columns: {missing}")

    out = df.copy()
    out["model_slug"] = slug_from_path(path)
    out["model_family"] = model_family_from_slug(out["model_slug"].iloc[0])
    out["source_prediction_file"] = str(path)

    if PARSE_STATUS_COL not in out.columns:
        out[PARSE_STATUS_COL] = np.where(out[PRED_COL].notna(), "ok_legacy_missing_parse_status", pd.NA)

    if RAW_OUTPUT_COL not in out.columns:
        out[RAW_OUTPUT_COL] = pd.NA

    return out


def load_all_predictions(pattern: str, min_models: int) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched: {pattern}")

    parts = [read_prediction_file(path) for path in paths]
    df = pd.concat(parts, ignore_index=True)

    n_models = df["model_slug"].nunique()
    if n_models < min_models:
        raise ValueError(
            f"Found only {n_models} model(s), but --min-models={min_models}. "
            f"Pattern: {pattern}"
        )

    duplicate = df.duplicated(["model_slug", ROW_ID_COL]).sum()
    if duplicate:
        raise ValueError(f"Found {duplicate} duplicate model/ROW_ID rows.")

    return df


def load_scores(scores_csv: str | Path) -> pd.DataFrame:
    path = Path(scores_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing Qwen score file: {path}")

    scores = pd.read_csv(path, low_memory=False)
    if AI_CONF_COL not in scores.columns:
        raise ValueError(f"{path} is missing {AI_CONF_COL!r}.")

    scores = scores.copy()
    scores[AI_CONF_COL] = pd.to_numeric(scores[AI_CONF_COL], errors="coerce")

    key_col = find_first_column(scores, ["job_key", JOB_KEY_COL, "JOB_CONTEXT_KEY"])
    if key_col is not None:
        out = scores[[key_col, AI_CONF_COL]].drop_duplicates(key_col)
        return out.rename(columns={key_col: JOB_KEY_COL})

    title_col = find_first_column(scores, ["JOB_TITLE", "job_title", "title"])
    soc_code_col = find_first_column(scores, ["SOC_CODE", "soc_code"])
    soc_title_col = find_first_column(scores, ["SOC_TITLE", "soc_title"])

    if title_col is None or soc_code_col is None or soc_title_col is None:
        raise ValueError(
            f"{path} must contain either job_key/{JOB_KEY_COL}, or title/SOC columns "
            f"from which a stable job context key can be built."
        )

    scores[JOB_KEY_COL] = [
        stable_job_key(t, sc, st)
        for t, sc, st in zip(scores[title_col], scores[soc_code_col], scores[soc_title_col])
    ]

    return scores[[JOB_KEY_COL, AI_CONF_COL]].drop_duplicates(JOB_KEY_COL)


def load_sampled_metadata(sampled_input: str | Path, scores_csv: str | Path) -> pd.DataFrame:
    sampled_path = Path(sampled_input)
    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled input CSV: {sampled_path}")

    meta = pd.read_csv(sampled_path, low_memory=False)

    missing = sorted(REQUIRED_SAMPLED_INPUT_COLUMNS - set(meta.columns))
    if missing:
        raise ValueError(f"{sampled_path} is missing required sampled-input columns: {missing}")

    meta = meta.copy()
    meta["_row_id_key"] = meta[ROW_ID_COL].astype(str)

    if JOB_KEY_COL not in meta.columns:
        meta[JOB_KEY_COL] = [
            stable_job_key(t, sc, st)
            for t, sc, st in zip(meta["JOB_TITLE"], meta["SOC_CODE"], meta["SOC_TITLE"])
        ]

    if BLOCK_COL not in meta.columns:
        meta[BLOCK_COL] = make_default_block_id(meta)

    if AI_CONF_COL not in meta.columns:
        scores = load_scores(scores_csv)
        meta = meta.merge(scores, on=JOB_KEY_COL, how="left")

    meta[AI_CONF_COL] = pd.to_numeric(meta[AI_CONF_COL], errors="coerce")

    if meta[AI_CONF_COL].isna().any():
        n_missing = int(meta[AI_CONF_COL].isna().sum())
        raise ValueError(f"Missing {AI_CONF_COL} for {n_missing} sampled rows after merge.")

    keep_cols = [
        "_row_id_key",
        ROW_ID_COL,
        JOB_KEY_COL,
        BLOCK_COL,
        AI_CONF_COL,
        "PW_WAGE_LEVEL",
        "WORKSITE_CITY",
    ]

    for optional_col in ["QWEN_AI_GROUP", "EMPLOYER_NAME"]:
        if optional_col in meta.columns:
            keep_cols.append(optional_col)

    keep_cols = [c for c in keep_cols if c in meta.columns]
    meta = meta[keep_cols].drop_duplicates("_row_id_key")

    if meta["_row_id_key"].duplicated().any():
        raise ValueError("Sampled metadata has duplicate ROW_ID values.")

    return meta


def merge_metadata(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["_row_id_key"] = out[ROW_ID_COL].astype(str)

    meta_cols = [c for c in meta.columns if c != ROW_ID_COL]
    out = out.merge(meta[meta_cols], on="_row_id_key", how="left", suffixes=("", "_meta"))

    for required_col in [JOB_KEY_COL, BLOCK_COL, AI_CONF_COL]:
        if required_col not in out.columns:
            raise ValueError(f"Missing {required_col} after metadata merge.")
        if out[required_col].isna().any():
            n = int(out[required_col].isna().sum())
            raise ValueError(f"Missing {required_col} for {n} model-row records after metadata merge.")

    return out.drop(columns=["_row_id_key"])


def coerce_and_filter_for_analysis(
    df: pd.DataFrame,
    *,
    min_salary_usd: int,
    max_salary_usd: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    out[ACTUAL_COL] = parse_money_series(out[ACTUAL_COL])
    out[PRED_COL] = parse_money_series(out[PRED_COL])
    out[AI_CONF_COL] = pd.to_numeric(out[AI_CONF_COL], errors="coerce").clip(0.0, 1.0)
    out["is_ai_binary"] = parse_bool_series(out[AI_BINARY_COL]).astype(int)

    out[PARSE_STATUS_COL] = out[PARSE_STATUS_COL].astype("string").fillna("missing_parse_status")
    out["parse_ok"] = out[PARSE_STATUS_COL].astype(str).isin(COMPLETED_PARSE_STATUSES)

    parse_summary = (
        out.groupby(["model_slug", PARSE_STATUS_COL], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["model_slug", PARSE_STATUS_COL])
    )

    raw_qc = (
        out.groupby("model_slug", as_index=False)
        .agg(
            model_family=("model_family", "first"),
            n_rows=(ROW_ID_COL, "size"),
            n_unique_rows=(ROW_ID_COL, "nunique"),
            parse_ok_rows=("parse_ok", "sum"),
            parse_ok_rate=("parse_ok", "mean"),
            missing_prediction_rows=(PRED_COL, lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
        )
    )

    bad_actual = out[
        out[ACTUAL_COL].isna()
        | (out[ACTUAL_COL] < min_salary_usd)
        | (out[ACTUAL_COL] > max_salary_usd)
    ]
    if not bad_actual.empty:
        raise ValueError(
            f"Found {len(bad_actual)} rows with missing/out-of-range {ACTUAL_COL} "
            f"outside [{min_salary_usd}, {max_salary_usd}]."
        )

    analysis = out[out["parse_ok"]].copy()
    analysis = analysis[analysis[PRED_COL].notna()].copy()

    bad_pred = analysis[
        (analysis[PRED_COL] < min_salary_usd)
        | (analysis[PRED_COL] > max_salary_usd)
    ]
    if not bad_pred.empty:
        examples = bad_pred[
            ["model_slug", ROW_ID_COL, PRED_COL, RAW_OUTPUT_COL, PARSE_STATUS_COL]
        ].head(20)
        raise ValueError(
            f"Found {len(bad_pred)} parsed predictions outside "
            f"[{min_salary_usd}, {max_salary_usd}] USD.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    if (analysis[ACTUAL_COL] <= 0).any():
        raise ValueError(f"{ACTUAL_COL} must be positive.")

    analysis[DIFF_USD_COL] = analysis[PRED_COL] - analysis[ACTUAL_COL]
    analysis[RATIO_COL] = analysis[PRED_COL] / analysis[ACTUAL_COL]
    analysis[PCT_POINTS_COL] = 100.0 * (analysis[PRED_COL] - analysis[ACTUAL_COL]) / analysis[ACTUAL_COL]
    analysis[LOG_RATIO_COL] = np.log(analysis[PRED_COL]) - np.log(analysis[ACTUAL_COL])

    analysis[BLOCK_COL] = analysis[BLOCK_COL].astype(str)
    analysis[ROW_ID_COL] = analysis[ROW_ID_COL].astype(str)
    analysis["model_slug"] = analysis["model_slug"].astype(str)

    return analysis, parse_summary, raw_qc


def prediction_vector_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    vectors: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    for model_slug, g in df.groupby("model_slug", sort=True):
        ordered = g.sort_values(ROW_ID_COL)
        row_ids = tuple(ordered[ROW_ID_COL].astype(str).tolist())
        preds = tuple(
            pd.to_numeric(ordered[PRED_COL], errors="coerce")
            .round(0)
            .astype("Int64")
            .astype(str)
            .tolist()
        )
        digest = hashlib.md5(
            ("\n".join(row_ids) + "\n---\n" + "\n".join(preds)).encode("utf-8")
        ).hexdigest()

        rows.append(
            {
                "model_slug": model_slug,
                "model_family": ordered["model_family"].iloc[0],
                "n_rows": len(ordered),
                "n_unique_rows": ordered[ROW_ID_COL].nunique(),
                "prediction_vector_hash": digest,
                "n_unique_predicted_salaries": int(ordered[PRED_COL].nunique()),
                "min_predicted_salary_usd": float(ordered[PRED_COL].min()),
                "max_predicted_salary_usd": float(ordered[PRED_COL].max()),
            }
        )
        vectors[model_slug] = (row_ids, preds)

    dup_rows: list[dict[str, object]] = []
    names = sorted(vectors)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            same_row_order = vectors[a][0] == vectors[b][0]
            same_predictions = vectors[a][1] == vectors[b][1]
            if same_row_order and same_predictions:
                dup_rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "model_a_family": model_family_from_slug(a),
                        "model_b_family": model_family_from_slug(b),
                        "same_row_order": True,
                        "same_prediction_vector": True,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(dup_rows)


def make_row_average(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        ROW_ID_COL,
        "JOB_TITLE",
        "SOC_CODE",
        "SOC_TITLE",
        "WORKSITE_CITY",
        "WORKSITE_STATE",
        "NAICS_CODE",
        "FULL_TIME_POSITION",
        "TOTAL_WORKER_POSITIONS",
        "PW_WAGE_LEVEL",
        ACTUAL_COL,
        AI_BINARY_COL,
        "is_ai_binary",
        AI_CONF_COL,
        JOB_KEY_COL,
        BLOCK_COL,
    ]
    existing_base = [c for c in base_cols if c in df.columns]
    base = df.drop_duplicates(ROW_ID_COL)[existing_base].copy()

    agg = (
        df.groupby(ROW_ID_COL, as_index=False)
        .agg(
            n_models=("model_slug", "nunique"),
            mean_predicted_salary_usd=(PRED_COL, "mean"),
            median_predicted_salary_usd=(PRED_COL, "median"),
            mean_overestimation_usd=(DIFF_USD_COL, "mean"),
            median_overestimation_usd=(DIFF_USD_COL, "median"),
            mean_overestimation_pct_points=(PCT_POINTS_COL, "mean"),
            median_overestimation_pct_points=(PCT_POINTS_COL, "median"),
            mean_prediction_to_actual_ratio=(RATIO_COL, "mean"),
            mean_log_prediction_minus_log_actual=(LOG_RATIO_COL, "mean"),
        )
    )

    return base.merge(agg, on=ROW_ID_COL, how="inner")


def welch_test_row_average(row_df: pd.DataFrame) -> pd.DataFrame:
    scipy_stats = import_stats()

    ai = row_df.loc[row_df["is_ai_binary"] == 1, "mean_overestimation_pct_points"].dropna()
    other = row_df.loc[row_df["is_ai_binary"] == 0, "mean_overestimation_pct_points"].dropna()

    if len(ai) < 2 or len(other) < 2:
        t = np.nan
        p = np.nan
        status = "not_estimable"
    else:
        res = scipy_stats.ttest_ind(ai, other, equal_var=False, nan_policy="omit")
        t = float(res.statistic)
        p = float(res.pvalue)
        status = "ok"

    return pd.DataFrame(
        [
            {
                "analysis": "secondary_row_average_welch_ai_vs_other",
                "outcome": "mean_overestimation_pct_points",
                "unit": "H1B row averaged across models",
                "mean_ai_pct_points": float(ai.mean()) if len(ai) else np.nan,
                "mean_other_pct_points": float(other.mean()) if len(other) else np.nan,
                "difference_ai_minus_other_pct_points": float(ai.mean() - other.mean()) if len(ai) and len(other) else np.nan,
                "median_ai_pct_points": float(ai.median()) if len(ai) else np.nan,
                "median_other_pct_points": float(other.median()) if len(other) else np.nan,
                "n_ai_rows": int(len(ai)),
                "n_other_rows": int(len(other)),
                "t_value": t,
                "p_value": p,
                "status": status,
            }
        ]
    )


def block_level_contrasts(row_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scipy_stats = import_stats()

    contrast_rows: list[dict[str, object]] = []

    for block_id, g in row_df.groupby(BLOCK_COL, sort=True):
        ai = g[g["is_ai_binary"] == 1]
        other = g[g["is_ai_binary"] == 0]

        if ai.empty or other.empty:
            continue

        ai_mean_pct = ai["mean_overestimation_pct_points"].mean()
        other_mean_pct = other["mean_overestimation_pct_points"].mean()
        ai_mean_usd = ai["mean_overestimation_usd"].mean()
        other_mean_usd = other["mean_overestimation_usd"].mean()

        contrast_rows.append(
            {
                BLOCK_COL: block_id,
                "n_ai_rows": int(len(ai)),
                "n_other_rows": int(len(other)),
                "mean_ai_overestimation_pct_points": float(ai_mean_pct),
                "mean_other_overestimation_pct_points": float(other_mean_pct),
                "delta_pct_points_ai_minus_other": float(ai_mean_pct - other_mean_pct),
                "median_ai_overestimation_pct_points": float(ai["mean_overestimation_pct_points"].median()),
                "median_other_overestimation_pct_points": float(other["mean_overestimation_pct_points"].median()),
                "mean_ai_overestimation_usd": float(ai_mean_usd),
                "mean_other_overestimation_usd": float(other_mean_usd),
                "delta_usd_ai_minus_other": float(ai_mean_usd - other_mean_usd),
                "mean_ai_confidence": float(ai[AI_CONF_COL].mean()),
                "mean_other_confidence": float(other[AI_CONF_COL].mean()),
                "harmonic_weight": float((len(ai) * len(other)) / (len(ai) + len(other))),
            }
        )

    contrasts = pd.DataFrame(contrast_rows)

    if contrasts.empty:
        summary = pd.DataFrame(
            [
                {
                    "analysis": "primary_block_level_contrast_ttest",
                    "outcome": "delta_pct_points_ai_minus_other",
                    "unit": "BLOCK_ID",
                    "status": "not_estimable_no_common_support_blocks",
                }
            ]
        )
        return contrasts, summary

    vals = contrasts["delta_pct_points_ai_minus_other"].dropna()

    if len(vals) < 2:
        t = np.nan
        p = np.nan
        status = "not_estimable_too_few_blocks"
    else:
        res = scipy_stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
        t = float(res.statistic)
        p = float(res.pvalue)
        status = "ok"

    n_nonzero = int((vals != 0).sum())
    if n_nonzero:
        sign = scipy_stats.binomtest(
            int((vals > 0).sum()),
            n=n_nonzero,
            p=0.5,
            alternative="two-sided",
        )
        sign_p = float(sign.pvalue)
    else:
        sign_p = np.nan

    summary = pd.DataFrame(
        [
            {
                "analysis": "primary_block_level_contrast_ttest",
                "outcome": "delta_pct_points_ai_minus_other",
                "unit": "BLOCK_ID",
                "mean_delta_pct_points_ai_minus_other": float(vals.mean()) if len(vals) else np.nan,
                "median_delta_pct_points_ai_minus_other": float(vals.median()) if len(vals) else np.nan,
                "std_delta_pct_points_ai_minus_other": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "n_common_support_blocks": int(len(vals)),
                "positive_blocks": int((vals > 0).sum()),
                "negative_blocks": int((vals < 0).sum()),
                "zero_blocks": int((vals == 0).sum()),
                "t_value": t,
                "p_value": p,
                "sign_test_p_value": sign_p,
                "status": status,
            }
        ]
    )

    return contrasts, summary


def per_model_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scipy_stats = import_stats()

    welch_rows: list[dict[str, object]] = []
    block_contrast_rows: list[dict[str, object]] = []
    block_summary_rows: list[dict[str, object]] = []

    for model_slug, g in df.groupby("model_slug", sort=True):
        family = g["model_family"].iloc[0]

        ai = g.loc[g["is_ai_binary"] == 1, PCT_POINTS_COL].dropna()
        other = g.loc[g["is_ai_binary"] == 0, PCT_POINTS_COL].dropna()

        if len(ai) >= 2 and len(other) >= 2:
            res = scipy_stats.ttest_ind(ai, other, equal_var=False, nan_policy="omit")
            t = float(res.statistic)
            p = float(res.pvalue)
            status = "ok"
        else:
            t = np.nan
            p = np.nan
            status = "not_estimable"

        welch_rows.append(
            {
                "analysis": "per_model_welch_ai_vs_other",
                "model_slug": model_slug,
                "model_family": family,
                "outcome": PCT_POINTS_COL,
                "mean_ai_pct_points": float(ai.mean()) if len(ai) else np.nan,
                "mean_other_pct_points": float(other.mean()) if len(other) else np.nan,
                "difference_ai_minus_other_pct_points": float(ai.mean() - other.mean()) if len(ai) and len(other) else np.nan,
                "n_ai_rows": int(len(ai)),
                "n_other_rows": int(len(other)),
                "t_value": t,
                "p_value": p,
                "status": status,
            }
        )

        for block_id, bg in g.groupby(BLOCK_COL, sort=True):
            bai = bg[bg["is_ai_binary"] == 1]
            bother = bg[bg["is_ai_binary"] == 0]

            if bai.empty or bother.empty:
                continue

            delta = bai[PCT_POINTS_COL].mean() - bother[PCT_POINTS_COL].mean()

            block_contrast_rows.append(
                {
                    "model_slug": model_slug,
                    "model_family": family,
                    BLOCK_COL: block_id,
                    "n_ai_rows": int(len(bai)),
                    "n_other_rows": int(len(bother)),
                    "mean_ai_overestimation_pct_points": float(bai[PCT_POINTS_COL].mean()),
                    "mean_other_overestimation_pct_points": float(bother[PCT_POINTS_COL].mean()),
                    "delta_pct_points_ai_minus_other": float(delta),
                    "mean_ai_overestimation_usd": float(bai[DIFF_USD_COL].mean()),
                    "mean_other_overestimation_usd": float(bother[DIFF_USD_COL].mean()),
                    "delta_usd_ai_minus_other": float(bai[DIFF_USD_COL].mean() - bother[DIFF_USD_COL].mean()),
                }
            )

    block_contrasts = pd.DataFrame(block_contrast_rows)

    if not block_contrasts.empty:
        for model_slug, g in block_contrasts.groupby("model_slug", sort=True):
            vals = g["delta_pct_points_ai_minus_other"].dropna()
            family = g["model_family"].iloc[0]

            if len(vals) >= 2:
                res = scipy_stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t = float(res.statistic)
                p = float(res.pvalue)
                status = "ok"
            else:
                t = np.nan
                p = np.nan
                status = "not_estimable"

            block_summary_rows.append(
                {
                    "analysis": "per_model_block_contrast_ttest",
                    "model_slug": model_slug,
                    "model_family": family,
                    "outcome": "delta_pct_points_ai_minus_other",
                    "mean_delta_pct_points_ai_minus_other": float(vals.mean()) if len(vals) else np.nan,
                    "median_delta_pct_points_ai_minus_other": float(vals.median()) if len(vals) else np.nan,
                    "n_common_support_blocks": int(len(vals)),
                    "positive_blocks": int((vals > 0).sum()),
                    "negative_blocks": int((vals < 0).sum()),
                    "t_value": t,
                    "p_value": p,
                    "status": status,
                }
            )

    return (
        pd.DataFrame(welch_rows),
        block_contrasts,
        pd.DataFrame(block_summary_rows),
    )


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_slug, g in df.groupby("model_slug", sort=True):
        ai = g[g["is_ai_binary"] == 1]
        other = g[g["is_ai_binary"] == 0]

        rows.append(
            {
                "model_slug": model_slug,
                "model_family": g["model_family"].iloc[0],
                "n_rows": int(len(g)),
                "n_ai_rows": int(len(ai)),
                "n_other_rows": int(len(other)),
                "mean_actual_usd": float(g[ACTUAL_COL].mean()),
                "mean_predicted_usd": float(g[PRED_COL].mean()),
                "mean_overestimation_usd": float(g[DIFF_USD_COL].mean()),
                "median_overestimation_usd": float(g[DIFF_USD_COL].median()),
                "mean_overestimation_pct_points": float(g[PCT_POINTS_COL].mean()),
                "median_overestimation_pct_points": float(g[PCT_POINTS_COL].median()),
                "mean_ai_overestimation_pct_points": float(ai[PCT_POINTS_COL].mean()) if len(ai) else np.nan,
                "mean_other_overestimation_pct_points": float(other[PCT_POINTS_COL].mean()) if len(other) else np.nan,
                "delta_ai_minus_other_pct_points": (
                    float(ai[PCT_POINTS_COL].mean() - other[PCT_POINTS_COL].mean())
                    if len(ai) and len(other)
                    else np.nan
                ),
                "mean_ai_overestimation_usd": float(ai[DIFF_USD_COL].mean()) if len(ai) else np.nan,
                "mean_other_overestimation_usd": float(other[DIFF_USD_COL].mean()) if len(other) else np.nan,
                "delta_ai_minus_other_usd": (
                    float(ai[DIFF_USD_COL].mean() - other[DIFF_USD_COL].mean())
                    if len(ai) and len(other)
                    else np.nan
                ),
                "corr_ai_confidence_overestimation_pct_points": (
                    float(g[[AI_CONF_COL, PCT_POINTS_COL]].corr().iloc[0, 1])
                    if g[AI_CONF_COL].nunique() > 1 and g[PCT_POINTS_COL].nunique() > 1
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def ai_group_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for value, g in row_df.groupby("is_ai_binary", sort=True):
        group = "AI" if int(value) == 1 else "Other"

        rows.append(
            {
                "group": group,
                "n_rows": int(len(g)),
                "mean_ai_confidence": float(g[AI_CONF_COL].mean()),
                "mean_actual_usd": float(g[ACTUAL_COL].mean()),
                "mean_predicted_salary_usd": float(g["mean_predicted_salary_usd"].mean()),
                "mean_overestimation_usd": float(g["mean_overestimation_usd"].mean()),
                "median_overestimation_usd": float(g["mean_overestimation_usd"].median()),
                "mean_overestimation_pct_points": float(g["mean_overestimation_pct_points"].mean()),
                "median_overestimation_pct_points": float(g["mean_overestimation_pct_points"].median()),
            }
        )

    return pd.DataFrame(rows)


def continuous_ai_confidence_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    scipy_stats = import_stats()

    d = row_df[[AI_CONF_COL, "mean_overestimation_pct_points"]].dropna().copy()

    if len(d) < 3 or d[AI_CONF_COL].nunique() < 2 or d["mean_overestimation_pct_points"].nunique() < 2:
        pearson_r = np.nan
        pearson_p = np.nan
        spearman_r = np.nan
        spearman_p = np.nan
        status = "not_estimable"
    else:
        pr = scipy_stats.pearsonr(d[AI_CONF_COL], d["mean_overestimation_pct_points"])
        sr = scipy_stats.spearmanr(d[AI_CONF_COL], d["mean_overestimation_pct_points"])
        pearson_r = float(pr.statistic)
        pearson_p = float(pr.pvalue)
        spearman_r = float(sr.statistic)
        spearman_p = float(sr.pvalue)
        status = "ok"

    return pd.DataFrame(
        [
            {
                "analysis": "row_average_continuous_ai_confidence_correlation",
                "outcome": "mean_overestimation_pct_points",
                "n_rows": int(len(d)),
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p_value": spearman_p,
                "status": status,
            }
        ]
    )


def make_report(
    *,
    out_dir: Path,
    config: AnalysisConfig,
    analysis_df: pd.DataFrame,
    row_average: pd.DataFrame,
    parse_summary: pd.DataFrame,
    raw_qc: pd.DataFrame,
    prediction_qc: pd.DataFrame,
    duplicate_vectors: pd.DataFrame,
    block_contrast_summary: pd.DataFrame,
    row_welch: pd.DataFrame,
    per_model_welch: pd.DataFrame,
    per_model_block_summary: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    ai_summary: pd.DataFrame,
    confidence_summary: pd.DataFrame,
) -> str:
    primary_text = "Primary block-level test was not estimable."

    if not block_contrast_summary.empty:
        r = block_contrast_summary.iloc[0]
        if r.get("status") == "ok":
            primary_text = (
                f"Primary block-level contrast: mean AI-minus-Other difference = "
                f"{r['mean_delta_pct_points_ai_minus_other']:.6g} percentage points, "
                f"t={r['t_value']:.6g}, p={format_p(r['p_value'])}, "
                f"across {int(r['n_common_support_blocks'])} common-support blocks."
            )
        else:
            primary_text = f"Primary block-level contrast status: {r.get('status')}"

    secondary_text = "Secondary row-average Welch test was not estimable."
    if not row_welch.empty:
        r = row_welch.iloc[0]
        if r.get("status") == "ok":
            secondary_text = (
                f"Secondary row-level Welch test on model-averaged predictions: "
                f"AI mean = {r['mean_ai_pct_points']:.6g} percentage points, "
                f"Other mean = {r['mean_other_pct_points']:.6g} percentage points, "
                f"difference = {r['difference_ai_minus_other_pct_points']:.6g} percentage points, "
                f"t={r['t_value']:.6g}, p={format_p(r['p_value'])}."
            )
        else:
            secondary_text = f"Secondary Welch status: {r.get('status')}"

    lines = [
        "# H-1B Salary Overestimation Analysis",
        "",
        "## Question",
        "Do LLM salary estimators overestimate AI-related H-1B jobs more than matched non-AI H-1B jobs?",
        "",
        "## Primary outcome",
        f"`{PCT_POINTS_COL} = 100 × ({PRED_COL} - {ACTUAL_COL}) / {ACTUAL_COL}`.",
        "The reported AI effect is therefore measured in percentage points of actual prevailing wage.",
        "",
        "## Primary test",
        "The primary test first averages predictions across models for each H-1B row, then computes AI-minus-Other overestimation contrasts within each sampler block, and finally runs a one-sample t-test over block-level contrasts.",
        "",
        "## Secondary test",
        "The secondary test runs Welch's t-test on row-level model-averaged percentage-point overestimation, comparing AI rows to Other rows.",
        "",
        "## Data",
        f"Model-row records used: {len(analysis_df):,}",
        f"Unique H-1B rows used: {analysis_df[ROW_ID_COL].nunique():,}",
        f"Prediction models used: {analysis_df['model_slug'].nunique():,}",
        f"Row-average table rows: {len(row_average):,}",
        f"Blocks in row-average table: {row_average[BLOCK_COL].nunique():,}",
        f"AI rows: {int((row_average['is_ai_binary'] == 1).sum()):,}",
        f"Other rows: {int((row_average['is_ai_binary'] == 0).sum()):,}",
        f"AI threshold: >= {config.ai_high_threshold:.2f}; Other threshold: <= {config.ai_low_threshold:.2f}",
        "",
        "## Main result",
        primary_text,
        "",
        "## Secondary result",
        secondary_text,
        "",
        "## Output files",
        "- `model_row_salary_overestimation_long.csv/parquet`: model × row data with percentage-point overestimation.",
        "- `row_average_salary_overestimation.csv`: one row per H-1B case after averaging across models.",
        "- `primary_block_level_contrasts.csv`: within-block AI-minus-Other contrasts.",
        "- `primary_block_level_contrast_summary.csv`: primary t-test over block contrasts.",
        "- `secondary_row_average_welch.csv`: row-average Welch test.",
        "- `per_model_welch_tests.csv`: per-model Welch tests.",
        "- `per_model_block_level_contrasts.csv`: per-model within-block contrasts.",
        "- `per_model_block_contrast_summary.csv`: per-model t-tests over block contrasts.",
        "- `model_salary_overestimation_summary.csv`: descriptive model summary.",
        "- `ai_group_salary_overestimation_summary.csv`: row-average group summary.",
        "- `continuous_ai_confidence_summary.csv`: correlation between AI confidence and overestimation.",
        "- `parse_status_summary.csv`: parsing/compliance summary.",
        "- `prediction_vector_qc.csv` and `identical_prediction_vectors.csv`: output-integrity checks.",
    ]

    if duplicate_vectors.empty:
        lines.extend(["", "## Prediction-output QC", "", "No identical prediction vectors were detected."])
    else:
        lines.extend(
            [
                "",
                "## Prediction-output QC",
                "",
                "WARNING: identical prediction vectors were detected. Inspect before making final claims.",
                "",
                duplicate_vectors.to_markdown(index=False),
            ]
        )

    sections = [
        ("Parse-status summary", parse_summary),
        ("Raw model-output QC", raw_qc),
        ("Primary block-level contrast summary", block_contrast_summary),
        ("Secondary row-average Welch test", row_welch),
        ("Continuous AI-confidence summary", confidence_summary),
        ("AI group descriptive summary", ai_summary),
        ("Model summary", model_summary_df),
        ("Per-model Welch tests", per_model_welch),
        ("Per-model block-contrast summary", per_model_block_summary),
    ]

    for title, table in sections:
        if table is not None and not table.empty:
            lines.extend(["", f"## {title}", "", table.to_markdown(index=False)])

    report = "\n".join(lines) + "\n"
    (out_dir / "salary_overestimation_simple_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig(
        predictions_glob=args.predictions_glob,
        sampled_input=args.sampled_input,
        scores_csv=args.scores_csv,
        ai_high_threshold=args.ai_high_threshold,
        ai_low_threshold=args.ai_low_threshold,
        min_salary_usd=args.min_salary_usd,
        max_salary_usd=args.max_salary_usd,
        min_models=args.min_models,
        primary_outcome=PCT_POINTS_COL,
        primary_test="one-sample t-test over block-level AI-minus-Other percentage-point contrasts",
    )

    raw_predictions = load_all_predictions(args.predictions_glob, min_models=args.min_models)
    metadata = load_sampled_metadata(args.sampled_input, args.scores_csv)
    merged = merge_metadata(raw_predictions, metadata)

    analysis_df, parse_summary, raw_qc = coerce_and_filter_for_analysis(
        merged,
        min_salary_usd=args.min_salary_usd,
        max_salary_usd=args.max_salary_usd,
    )

    preferred_order = [
        "model_slug",
        "model_family",
        ROW_ID_COL,
        JOB_KEY_COL,
        "JOB_TITLE",
        "SOC_CODE",
        "SOC_TITLE",
        "WORKSITE_CITY",
        "WORKSITE_STATE",
        "NAICS_CODE",
        "FULL_TIME_POSITION",
        "TOTAL_WORKER_POSITIONS",
        "PW_WAGE_LEVEL",
        BLOCK_COL,
        AI_BINARY_COL,
        "is_ai_binary",
        AI_CONF_COL,
        ACTUAL_COL,
        PRED_COL,
        DIFF_USD_COL,
        PCT_POINTS_COL,
        RATIO_COL,
        LOG_RATIO_COL,
        PARSE_STATUS_COL,
        RAW_OUTPUT_COL,
        "source_prediction_file",
    ]
    ordered_cols = [c for c in preferred_order if c in analysis_df.columns]
    ordered_cols += [c for c in analysis_df.columns if c not in ordered_cols]
    analysis_df = analysis_df[ordered_cols].copy()

    prediction_qc, duplicate_vectors = prediction_vector_qc(analysis_df)

    if args.fail_on_identical_prediction_vectors and not duplicate_vectors.empty:
        raise ValueError(
            "Identical prediction vectors detected. Rerun without "
            "--fail-on-identical-prediction-vectors to write diagnostics."
        )

    row_average = make_row_average(analysis_df)

    block_contrasts, block_contrast_summary = block_level_contrasts(row_average)
    row_welch = welch_test_row_average(row_average)
    per_model_welch, per_model_block_contrasts, per_model_block_summary = per_model_tests(analysis_df)

    model_summary_df = model_summary(analysis_df)
    ai_summary = ai_group_summary(row_average)
    confidence_summary = continuous_ai_confidence_summary(row_average)

    write_table(
        analysis_df,
        out_dir / "model_row_salary_overestimation_long.csv",
        out_dir / "model_row_salary_overestimation_long.parquet",
    )
    write_table(row_average, out_dir / "row_average_salary_overestimation.csv")
    write_table(block_contrasts, out_dir / "primary_block_level_contrasts.csv")
    write_table(block_contrast_summary, out_dir / "primary_block_level_contrast_summary.csv")
    write_table(row_welch, out_dir / "secondary_row_average_welch.csv")
    write_table(per_model_welch, out_dir / "per_model_welch_tests.csv")
    write_table(per_model_block_contrasts, out_dir / "per_model_block_level_contrasts.csv")
    write_table(per_model_block_summary, out_dir / "per_model_block_contrast_summary.csv")
    write_table(model_summary_df, out_dir / "model_salary_overestimation_summary.csv")
    write_table(ai_summary, out_dir / "ai_group_salary_overestimation_summary.csv")
    write_table(confidence_summary, out_dir / "continuous_ai_confidence_summary.csv")
    write_table(parse_summary, out_dir / "parse_status_summary.csv")
    write_table(raw_qc, out_dir / "raw_model_output_qc.csv")
    write_table(prediction_qc, out_dir / "prediction_vector_qc.csv")
    write_table(duplicate_vectors, out_dir / "identical_prediction_vectors.csv")

    meta = {
        "config": asdict(config),
        "n_model_row_records_used": int(len(analysis_df)),
        "n_unique_rows_used": int(analysis_df[ROW_ID_COL].nunique()),
        "n_models_used": int(analysis_df["model_slug"].nunique()),
        "models": sorted(analysis_df["model_slug"].astype(str).unique().tolist()),
        "n_row_average_records": int(len(row_average)),
        "n_blocks": int(row_average[BLOCK_COL].nunique()),
        "n_common_support_blocks": int(len(block_contrasts)),
        "n_identical_prediction_vector_pairs": int(len(duplicate_vectors)),
    }
    (out_dir / "salary_overestimation_simple_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    report = make_report(
        out_dir=out_dir,
        config=config,
        analysis_df=analysis_df,
        row_average=row_average,
        parse_summary=parse_summary,
        raw_qc=raw_qc,
        prediction_qc=prediction_qc,
        duplicate_vectors=duplicate_vectors,
        block_contrast_summary=block_contrast_summary,
        row_welch=row_welch,
        per_model_welch=per_model_welch,
        per_model_block_summary=per_model_block_summary,
        model_summary_df=model_summary_df,
        ai_summary=ai_summary,
        confidence_summary=confidence_summary,
    )

    print(report)


if __name__ == "__main__":
    main()

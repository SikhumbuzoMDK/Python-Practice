"""
calculations.py
Implements business calculation logic mapped directly from ANALYSIS.xlsx
"""
import numpy as np
import pandas as pd

# Core Eligible Payout Types from ANALYSIS.xlsx
CORE_PAYOUT_TYPES = ["NEW PAYOUT", "PROGRESS DRAW", "RE-ADVANCE", "RESTRUCTURE_CALC"]

# Regional Weekly Targets from Ranking / Product Targets sheet
DEFAULT_WEEKLY_TARGETS = {
    "INLAND": 58_750_000.0,
    "WESTERN CAPE": 49_326_923.08,
    "KZN": 61_250_000.0,
    "EASTERN CAPE": 13_846_153.85,
    "GAUTENG SOUTH WEST": 16_923_076.92,
    "GAUTENG EAST": 25_576_923.08,
    "GAUTENG KLIPRIVIER": 5_384_615.38,
    "GREATER SANDTON": 27_692_307.69,
    "GAUTENG TSHWANE": 20_480_769.23,
    "MIDRAND": 8_653_846.15,
    "GAUTENG WEALTH": 20_096_153.85,
    "VIRTUAL CHANNEL": 0.0,
}

# Regional Super-Region Mapping
REGION_SUPER_REGION_MAP = {
    "INLAND": "INLAND",
    "WESTERN CAPE": "SUPERCAPE",
    "EASTERN CAPE": "SUPERCAPE",
    "CAPE INLAND": "SUPERCAPE",
    "KZN": "KZN",
    "GAUTENG EAST": "GAUTENG SOUTH AND CENTRAL",
    "GAUTENG SOUTH WEST": "GAUTENG SOUTH AND CENTRAL",
    "GAUTENG KLIPRIVIER": "GAUTENG SOUTH AND CENTRAL",
    "GREATER SANDTON": "GAUTENG NORTH",
    "GAUTENG TSHWANE": "GAUTENG NORTH",
    "MIDRAND": "GAUTENG NORTH",
    "GAUTENG WEALTH": "GAUTENG WEALTH",
    "VIRTUAL CHANNEL": "VIRTUAL CHANNEL",
}


def resolve_column(df: pd.DataFrame, candidates: list, fallback: str = "") -> str:
    """Finds matching column irrespective of case, spaces, or underscores."""
    col_lookup = {str(c).upper().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        clean_cand = cand.upper().replace(" ", "").replace("_", "")
        if clean_cand in col_lookup:
            return col_lookup[clean_cand]
    return fallback


def filter_by_horizon(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Filters dataset according to WTD, MTD, or YTD criteria."""
    df_copy = df.copy()

    max_wk_col = resolve_column(df_copy, ["MAX_WEEK", "MAXWEEK"])
    month_filter_col = resolve_column(df_copy, ["LATEST_MONTH_FILTER", "LATESTMONTHFILTER"])
    wk_col = resolve_column(df_copy, ["WEEK", "FISCAL_WEEK", "FSCL_WEEK"])
    m_col = resolve_column(df_copy, ["MONTH", "FISCAL_MONTH"])

    if horizon == "WTD":
        if max_wk_col and max_wk_col in df_copy.columns:
            filtered = df_copy[df_copy[max_wk_col].astype(str).str.strip().str.upper() == "Y"]
            if not filtered.empty:
                return filtered
        if wk_col and wk_col in df_copy.columns:
            try:
                max_wk = pd.to_numeric(df_copy[wk_col], errors="coerce").max()
                return df_copy[df_copy[wk_col] == max_wk]
            except Exception:
                pass

    elif horizon == "MTD":
        if month_filter_col and month_filter_col in df_copy.columns:
            filtered = df_copy[df_copy[month_filter_col].astype(str).str.strip().str.upper() == "Y"]
            if not filtered.empty:
                return filtered
        if m_col and m_col in df_copy.columns:
            try:
                max_m = pd.to_numeric(df_copy[m_col], errors="coerce").max()
                return df_copy[df_copy[m_col] == max_m]
            except Exception:
                pass

    # YTD: Return full dataset
    return df_copy


def get_target_multiplier(horizon: str) -> float:
    """Returns multiplier for Weekly Target based on horizon."""
    if horizon == "WTD":
        return 1.0
    elif horizon == "MTD":
        return 4.0
    elif horizon == "YTD":
        return 24.0
    return 1.0


def calculate_sales_indicators(df_horizon: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Computes Sales Indicators Consolidation Table per region."""
    amount_col = resolve_column(
        df_horizon,
        ["SUM_OF_TXN_AMNT", "SUM_OF_TRN_AMNT", "SUMOFTRNAMNT", "TXN_AMNT", "AMOUNT", "PAYOUT"],
        fallback=""
    )
    if not amount_col or amount_col not in df_horizon.columns:
        df_horizon = df_horizon.copy()
        amount_col = "SUM_OF_TXN_AMNT"
        df_horizon[amount_col] = 0.0

    region_col = resolve_column(df_horizon, ["GRPD_REGION", "DM_REGION", "REGION"], fallback="")
    payout_col = resolve_column(df_horizon, ["PAYOUT_TYPE", "TRANSACTION_TYPE", "TYPE"], fallback="")
    growth_col = resolve_column(df_horizon, ["GROWTH_SEG_IND", "GROWTH"], fallback="")

    regions = list(DEFAULT_WEEKLY_TARGETS.keys())
    multiplier = get_target_multiplier(horizon)

    rows = []
    for reg in regions:
        if region_col and region_col in df_horizon.columns:
            reg_df = df_horizon[df_horizon[region_col].astype(str).str.strip().str.upper() == reg]
        else:
            reg_df = df_horizon

        if payout_col and payout_col in reg_df.columns:
            core_mask = reg_df[payout_col].astype(str).str.strip().str.upper().isin(CORE_PAYOUT_TYPES)
        else:
            core_mask = pd.Series([True] * len(reg_df), index=reg_df.index)

        # Growth Segment
        if growth_col and growth_col in reg_df.columns:
            growth_mask = core_mask & (reg_df[growth_col].astype(str).str.strip().str.upper() == "Y")
            growth_val = pd.to_numeric(reg_df[growth_mask][amount_col], errors="coerce").fillna(0.0).sum()
        else:
            growth_val = 0.0

        # Restructures
        if payout_col and payout_col in reg_df.columns:
            restr_mask = reg_df[payout_col].astype(str).str.strip().str.upper() == "RESTRUCTURE_CALC"
            restr_val = pd.to_numeric(reg_df[restr_mask][amount_col], errors="coerce").fillna(0.0).sum()
        else:
            restr_val = 0.0

        # Re-advances
        if payout_col and payout_col in reg_df.columns:
            readv_mask = reg_df[payout_col].astype(str).str.strip().str.upper() == "RE-ADVANCE"
            readv_val = pd.to_numeric(reg_df[readv_mask][amount_col], errors="coerce").fillna(0.0).sum()
        else:
            readv_val = 0.0

        # Total Core Paid Out
        total_payout = pd.to_numeric(reg_df[core_mask][amount_col], errors="coerce").fillna(0.0).sum()

        # Other Segment = Total - (Growth + Restructures + Re-advances)
        other_val = max(0.0, total_payout - (growth_val + restr_val + readv_val))

        weekly_target = DEFAULT_WEEKLY_TARGETS.get(reg, 0.0)
        target = weekly_target * multiplier
        pct_achieved = (total_payout / target * 100.0) if target > 0 else 0.0

        rows.append({
            "Region": reg,
            "Growth Payouts": growth_val,
            "Restructures": restr_val,
            "Re-advance Payouts": readv_val,
            "Other Segment": other_val,
            "Total Paid Out": total_payout,
            f"{horizon} Target": target,
            "% Target Achieved": pct_achieved,
        })

    summary_df = pd.DataFrame(rows)

    total_row = {
        "Region": "TOTAL",
        "Growth Payouts": summary_df["Growth Payouts"].sum(),
        "Restructures": summary_df["Restructures"].sum(),
        "Re-advance Payouts": summary_df["Re-advance Payouts"].sum(),
        "Other Segment": summary_df["Other Segment"].sum(),
        "Total Paid Out": summary_df["Total Paid Out"].sum(),
        f"{horizon} Target": summary_df[f"{horizon} Target"].sum(),
        "% Target Achieved": (
            summary_df["Total Paid Out"].sum() / summary_df[f"{horizon} Target"].sum() * 100.0
            if summary_df[f"{horizon} Target"].sum() > 0
            else 0.0
        ),
    }
    return pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)


def calculate_product_group_analysis(df_horizon: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Computes Product Group Breakdown per Region."""
    amount_col = resolve_column(
        df_horizon,
        ["SUM_OF_TXN_AMNT", "SUM_OF_TRN_AMNT", "SUMOFTRNAMNT", "TXN_AMNT", "AMOUNT", "PAYOUT"],
        fallback=""
    )
    if not amount_col or amount_col not in df_horizon.columns:
        df_horizon = df_horizon.copy()
        amount_col = "SUM_OF_TXN_AMNT"
        df_horizon[amount_col] = 0.0

    region_col = resolve_column(df_horizon, ["GRPD_REGION", "DM_REGION", "REGION"], fallback="")
    payout_col = resolve_column(df_horizon, ["PAYOUT_TYPE", "TRANSACTION_TYPE", "TYPE"], fallback="")
    prod_col = resolve_column(df_horizon, ["PRODUCT_GROUP", "PRODUCT", "PRODUCT_NAME"], fallback="")
    hogan_col = resolve_column(df_horizon, ["HOGAN_SEGMENT", "SEGMENT"], fallback="")

    regions = list(DEFAULT_WEEKLY_TARGETS.keys())
    multiplier = get_target_multiplier(horizon)

    rows = []
    for reg in regions:
        if region_col and region_col in df_horizon.columns:
            reg_df = df_horizon[df_horizon[region_col].astype(str).str.strip().str.upper() == reg]
        else:
            reg_df = df_horizon

        payout_s = reg_df[payout_col].astype(str).str.strip().str.upper() if payout_col and payout_col in reg_df.columns else pd.Series(["NEW PAYOUT"] * len(reg_df), index=reg_df.index)
        prod_s = reg_df[prod_col].astype(str).str.strip().str.upper() if prod_col and prod_col in reg_df.columns else pd.Series(["OS"] * len(reg_df), index=reg_df.index)
        hogan_s = reg_df[hogan_col].astype(str).str.strip().str.upper() if hogan_col and hogan_col in reg_df.columns else pd.Series([""] * len(reg_df), index=reg_df.index)

        amt_s = pd.to_numeric(reg_df[amount_col], errors="coerce").fillna(0.0)

        # 1. OS
        os_mask = payout_s.isin(CORE_PAYOUT_TYPES) & (prod_s == "OS")
        os_val = amt_s[os_mask].sum()

        # 2. IPRE
        ipre_mask = payout_s.isin(CORE_PAYOUT_TYPES) & (prod_s == "IPRE")
        ipre_val = amt_s[ipre_mask].sum()

        # 3. WEALTH
        wealth_mask = payout_s.isin(CORE_PAYOUT_TYPES) & (prod_s == "WEALTH")
        wealth_val = amt_s[wealth_mask].sum()

        # 4. ISLAMIC
        islamic_mask = payout_s.isin(CORE_PAYOUT_TYPES) & (prod_s.isin(["ISLAMIC", "SHARIAH", "SHARIA"]))
        islamic_val = amt_s[islamic_mask].sum()

        # 5. PSB (Includes PRE-PAID)
        psb_payout_types = CORE_PAYOUT_TYPES + ["PRE-PAID"]
        psb_mask = payout_s.isin(psb_payout_types) & ((prod_s == "PSB") | (hogan_s == "PUBLIC SECTOR"))
        psb_val = amt_s[psb_mask].sum()

        total_product_val = os_val + ipre_val + wealth_val + islamic_val + psb_val
        weekly_budget = DEFAULT_WEEKLY_TARGETS.get(reg, 0.0)
        target = weekly_budget * multiplier
        pct_achieved = (total_product_val / target * 100.0) if target > 0 else 0.0

        rows.append({
            "Region": reg,
            "OS (Owner Serviced)": os_val,
            "IPRE": ipre_val,
            "Wealth": wealth_val,
            "Islamic": islamic_val,
            "PSB (Public Sector)": psb_val,
            "Total Payout": total_product_val,
            f"{horizon} Target": target,
            "% Achieved": pct_achieved,
        })

    prod_df = pd.DataFrame(rows)
    total_row = {
        "Region": "TOTAL",
        "OS (Owner Serviced)": prod_df["OS (Owner Serviced)"].sum(),
        "IPRE": prod_df["IPRE"].sum(),
        "Wealth": prod_df["Wealth"].sum(),
        "Islamic": prod_df["Islamic"].sum(),
        "PSB (Public Sector)": prod_df["PSB (Public Sector)"].sum(),
        "Total Payout": prod_df["Total Payout"].sum(),
        f"{horizon} Target": prod_df[f"{horizon} Target"].sum(),
        "% Achieved": (
            prod_df["Total Payout"].sum() / prod_df[f"{horizon} Target"].sum() * 100.0
            if prod_df[f"{horizon} Target"].sum() > 0
            else 0.0
        ),
    }
    return pd.concat([prod_df, pd.DataFrame([total_row])], ignore_index=True)


def generate_mock_data() -> pd.DataFrame:
    """Generates synthetic benchmark data matching the exact workbook schema."""
    np.random.seed(42)
    regions = list(DEFAULT_WEEKLY_TARGETS.keys())
    payout_types = ["NEW PAYOUT", "PROGRESS DRAW", "RE-ADVANCE", "RESTRUCTURE_CALC", "PRE-PAID", "VAL FEE1", "MANAG FEE"]
    dealmakers = [
        ("JACQUES BEDEKER", "GREATER SANDTON", "OS"),
        ("STEVE KOEHORST", "WESTERN CAPE", "IPRE"),
        ("GAVIN BREMER", "WESTERN CAPE", "IPRE"),
        ("SAABERA MOTARA", "GAUTENG EAST", "OS"),
        ("CAROLLIZE LAING", "EASTERN CAPE", "WEALTH"),
        ("STEVEN PYPER", "KZN", "OS"),
        ("AMAR SINGH", "GAUTENG KLIPRIVIER", "IPRE"),
        ("DALE ROSENBERG", "GAUTENG KLIPRIVIER", "IPRE"),
        ("LOURENS SWANEPOEL", "INLAND", "IPRE"),
        ("JP DU TOIT", "GREATER SANDTON", "WEALTH"),
    ]

    records = []
    for week in range(1, 25):
        is_max_week = "Y" if week == 24 else "N"
        month = int(np.ceil(week / 4))
        is_latest_month = "Y" if month == 6 else "N"

        for dm_name, dm_region, dm_prod in dealmakers:
            num_txns = np.random.randint(1, 4)
            for _ in range(num_txns):
                ptype = np.random.choice(payout_types, p=[0.45, 0.25, 0.15, 0.07, 0.04, 0.02, 0.02])
                amount = np.random.uniform(500_000, 35_000_000)
                growth_ind = "Y" if np.random.rand() > 0.65 else "N"
                cas_check = "TRUE" if np.random.rand() > 0.5 else "#N/A"
                hogan = "PUBLIC SECTOR" if dm_prod == "PSB" else "COMMERCIAL"

                records.append({
                    "WEEK": week,
                    "MONTH": month,
                    "MAX_WEEK": is_max_week,
                    "LATEST_MONTH_FILTER": is_latest_month,
                    "DM_NAME": dm_name,
                    "GRPD_REGION": dm_region,
                    "PRODUCT_GROUP": dm_prod,
                    "PAYOUT_TYPE": ptype,
                    "GROWTH_SEG_IND": growth_ind,
                    "CAS_CHECK": cas_check,
                    "HOGAN_SEGMENT": hogan,
                    "SUM_OF_TXN_AMNT": amount,
                })

    return pd.DataFrame(records)
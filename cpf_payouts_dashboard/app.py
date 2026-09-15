"""
app.py - CPF & CAS Payouts Dashboard
Styled to match the Power BI Report & Excel Theme
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from calculations import (
    CORE_PAYOUT_TYPES,
    DEFAULT_WEEKLY_TARGETS,
    REGION_SUPER_REGION_MAP,
    filter_by_horizon,
    calculate_sales_indicators,
    calculate_product_group_analysis,
    generate_mock_data,
    get_target_multiplier,
    resolve_column,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CPF & CAS Payouts Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Corporate Power BI Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Canvas */
    .main {
        background-color: #F4F7F9;
    }
    
    /* Top Banner */
    .dashboard-header {
        background: linear-gradient(135deg, #002D38 0%, #004D5A 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 45, 56, 0.15);
    }
    .dashboard-header h1 {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: #99D5DC;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }

    /* KPI Metric Cards (Power BI Card Visual) */
    .pbi-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px 18px;
        border-top: 4px solid #00838F;
        border-left: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    .pbi-card-accent-growth { border-top-color: #10B981; }
    .pbi-card-accent-restr { border-top-color: #D97706; }
    .pbi-card-accent-readv { border-top-color: #0F766E; }
    .pbi-card-accent-target { border-top-color: #64748B; }

    .pbi-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .pbi-card-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #002D38;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    .pbi-card-sub {
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 600;
    }
    .tag-positive {
        color: #059669;
        background-color: #ECFDF5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .tag-negative {
        color: #DC2626;
        background-color: #FEF2F2;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Section Subheadings */
    .section-title {
        color: #002D38;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #FFFFFF;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 18px;
        color: #475569;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00838F !important;
        color: #FFFFFF !important;
        border-color: #00838F !important;
    }

    /* Dataframe table border and header */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Corporate Palette Settings
# ---------------------------------------------------------
COLOR_PRIMARY_TEAL = "#00838F"
COLOR_NAVY_DARK = "#002D38"
COLOR_TARGET_GRAY = "#CBD5E1"
COLOR_GROWTH_GREEN = "#10B981"
COLOR_RESTRUCTURE_AMBER = "#D97706"
COLOR_READVANCE_TEAL = "#0F766E"
COLOR_OTHER_BLUE = "#0284C7"
COLOR_TARGET_LINE = "#E11D48"

PRODUCT_COLOR_MAP = {
    "OS (Owner Serviced)": "#00838F",
    "IPRE": "#10B981",
    "Wealth": "#0F766E",
    "Islamic": "#D97706",
    "PSB (Public Sector)": "#6366F1",
}


def apply_pbi_chart_layout(fig, height=390):
    """Applies clean Power BI visual styling to Plotly charts."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, sans-serif", color="#334155", size=12),
        margin=dict(l=30, r=20, t=50, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)"
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#CBD5E1",
            tickcolor="#CBD5E1",
            tickfont=dict(size=11, color="#475569")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color="#475569")
        )
    )
    return fig


# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
@st.cache_data
def load_excel_file(file, sheet_name=None):
    xl = pd.ExcelFile(file)
    sheets = xl.sheet_names
    chosen_sheet = sheet_name

    if not chosen_sheet or chosen_sheet not in sheets:
        for pref in ["YTD DATA", "Data", "Validate", "Region Monthly View"]:
            if pref in sheets:
                chosen_sheet = pref
                break
        if not chosen_sheet:
            chosen_sheet = sheets[0]

    df = pd.read_excel(file, sheet_name=chosen_sheet)
    return df, sheets


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown(f"""
<div style="background-color:#002D38; padding:14px; border-radius:8px; margin-bottom:15px; text-align:center;">
    <h3 style="color:#00A3AD; margin:0; font-size:1.15rem; font-weight:700;">CPF & CAS PORTAL</h3>
    <span style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase;">Commercial Payouts Analytics</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Updated_MTD payouts / Test.xlsx",
    type=["xlsx", "xls", "csv"],
    help="Upload your workbook or explore with embedded benchmark data."
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        xl_probe = pd.ExcelFile(uploaded_file)
        selected_sheet = st.sidebar.selectbox("Active Sheet", xl_probe.sheet_names)
        raw_df, _ = load_excel_file(uploaded_file, selected_sheet)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    raw_df = generate_mock_data()
    st.sidebar.info("Using Benchmark Dataset (Weeks 1 to 24)")

st.sidebar.markdown("---")
st.sidebar.subheader("Horizon Window")
horizon = st.sidebar.radio(
    "Performance Period:",
    options=["WTD", "MTD", "YTD"],
    index=0,
    help="WTD = Latest Max Week | MTD = Latest Month | YTD = Full Fiscal Period"
)

st.sidebar.markdown("---")
all_super_regions = ["ALL"] + sorted(list(set(REGION_SUPER_REGION_MAP.values())))
selected_super = st.sidebar.selectbox("Super Region", all_super_regions)

# ---------------------------------------------------------
# Filtering & Calculation Core
# ---------------------------------------------------------
amount_col = resolve_column(
    raw_df,
    ["SUM_OF_TXN_AMNT", "SUM_OF_TRN_AMNT", "SUMOFTRNAMNT", "TXN_AMNT", "AMOUNT", "PAYOUT"],
    fallback=""
)
if not amount_col or amount_col not in raw_df.columns:
    num_cols = raw_df.select_dtypes(include=[np.number]).columns
    amount_col = num_cols[0] if len(num_cols) > 0 else raw_df.columns[0]

region_col = resolve_column(raw_df, ["GRPD_REGION", "DM_REGION", "REGION"], fallback="")
payout_col = resolve_column(raw_df, ["PAYOUT_TYPE", "TRANSACTION_TYPE", "TYPE"], fallback="")
dm_col = resolve_column(raw_df, ["DM_NAME", "DEALMAKER", "NAME"], fallback="")
week_col = resolve_column(raw_df, ["WEEK", "FISCAL_WEEK", "FSCL_WEEK"], fallback="")

filtered_horizon_df = filter_by_horizon(raw_df, horizon)

if selected_super != "ALL" and region_col and region_col in filtered_horizon_df.columns:
    allowed_regions = [r for r, s in REGION_SUPER_REGION_MAP.items() if s == selected_super]
    filtered_horizon_df = filtered_horizon_df[
        filtered_horizon_df[region_col].astype(str).str.strip().str.upper().isin(allowed_regions)
    ]

sales_df = calculate_sales_indicators(filtered_horizon_df, horizon)
prod_df = calculate_product_group_analysis(filtered_horizon_df, horizon)

# Totals Row
total_sales_row = sales_df[sales_df["Region"] == "TOTAL"].iloc[0]
total_payout = total_sales_row["Total Paid Out"]
total_target = total_sales_row[f"{horizon} Target"]
achieve_pct = total_sales_row["% Target Achieved"]
growth_total = total_sales_row["Growth Payouts"]
restr_total = total_sales_row["Restructures"]
readv_total = total_sales_row["Re-advance Payouts"]

# ---------------------------------------------------------
# Executive Top Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <h1>CPF & CAS Payouts Dashboard</h1>
    <p>Performance View: <b>{horizon}</b> &nbsp;|&nbsp; Super Region: <b>{selected_super}</b> &nbsp;|&nbsp; Targets based on <b>ANALYSIS.xlsx</b> standard specs</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top KPI Metric Cards (Power BI Card Visuals)
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

badge_class = "tag-positive" if achieve_pct >= 100 else "tag-negative"

with kpi1:
    st.markdown(f"""
    <div class="pbi-card">
        <div class="pbi-card-title">{horizon} Total Payouts</div>
        <div class="pbi-card-val">R {total_payout:,.0f}</div>
        <div class="pbi-card-sub"><span class="{badge_class}">{achieve_pct:.1f}% of Target</span></div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="pbi-card pbi-card-accent-target">
        <div class="pbi-card-title">{horizon} Target</div>
        <div class="pbi-card-val">R {total_target:,.0f}</div>
        <div class="pbi-card-sub" style="color:#64748B;">Target Multiplier: x{get_target_multiplier(horizon):.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="pbi-card pbi-card-accent-growth">
        <div class="pbi-card-title">Growth Segment</div>
        <div class="pbi-card-val">R {growth_total:,.0f}</div>
        <div class="pbi-card-sub" style="color:#059669;">{(growth_total/total_payout*100 if total_payout>0 else 0):.1f}% of Deals</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="pbi-card pbi-card-accent-restr">
        <div class="pbi-card-title">Restructures</div>
        <div class="pbi-card-val">R {restr_total:,.0f}</div>
        <div class="pbi-card-sub" style="color:#D97706;">{(restr_total/total_payout*100 if total_payout>0 else 0):.1f}% of Deals</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="pbi-card pbi-card-accent-readv">
        <div class="pbi-card-title">Re-Advances</div>
        <div class="pbi-card-val">R {readv_total:,.0f}</div>
        <div class="pbi-card-sub" style="color:#0F766E;">{(readv_total/total_payout*100 if total_payout>0 else 0):.1f}% of Deals</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Dashboard Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Regional Sales Indicators",
    "🏢 Product Group (CAS) Breakdown",
    "🏆 Dealmaker Ranking & Movements",
    "📈 Weekly Payout Tracker & Trends",
])

# ---------------------------------------------------------
# TAB 1: Regional Sales Indicators Consolidation
# ---------------------------------------------------------
with tabs[0]:
    st.markdown(f'<div class="section-title">Sales Indicators Consolidation — {horizon}</div>', unsafe_allow_html=True)
    st.caption("Tracking Actual Payouts across Growth, Restructures, and Re-advances against regional targets.")

    chart_data = sales_df[sales_df["Region"] != "TOTAL"].copy()

    col_c1, col_c2 = st.columns([1.1, 0.9])

    with col_c1:
        # Comparison Bar Chart: Payouts vs Target
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=chart_data["Region"],
            y=chart_data["Total Paid Out"],
            name="Actual Payouts",
            marker_color=COLOR_PRIMARY_TEAL
        ))
        fig_bar.add_trace(go.Bar(
            x=chart_data["Region"],
            y=chart_data[f"{horizon} Target"],
            name=f"{horizon} Target",
            marker_color=COLOR_TARGET_GRAY
        ))
        fig_bar.update_layout(
            title=f"<b>Total Paid Out vs {horizon} Target</b>",
            barmode="group",
            xaxis_tickangle=-40
        )
        apply_pbi_chart_layout(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_c2:
        # Deal Breakdown Stacked Bar Chart
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Growth Payouts"], name="Growth", marker_color=COLOR_GROWTH_GREEN))
        fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Restructures"], name="Restructures", marker_color=COLOR_RESTRUCTURE_AMBER))
        fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Re-advance Payouts"], name="Re-advance", marker_color=COLOR_READVANCE_TEAL))
        fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Other Segment"], name="Other Segment", marker_color=COLOR_OTHER_BLUE))
        fig_stack.update_layout(
            title=f"<b>Deal Composition Breakdown ({horizon})</b>",
            barmode="stack",
            xaxis_tickangle=-40
        )
        apply_pbi_chart_layout(fig_stack, height=380)
        st.plotly_chart(fig_stack, use_container_width=True)

    # Table formatted to match Power BI matrix view
    formatted_sales = sales_df.copy()
    for col in ["Growth Payouts", "Restructures", "Re-advance Payouts", "Other Segment", "Total Paid Out", f"{horizon} Target"]:
        formatted_sales[col] = formatted_sales[col].apply(lambda x: f"R {x:,.0f}")
    formatted_sales["% Target Achieved"] = formatted_sales["% Target Achieved"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(formatted_sales, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 2: Product Group (CAS) Analysis
# ---------------------------------------------------------
with tabs[1]:
    st.markdown(f'<div class="section-title">Product Group (CAS) Breakdown — {horizon}</div>', unsafe_allow_html=True)
    st.caption("Decomposition across OS, IPRE, Wealth, Islamic (Shariah), and PSB (Public Sector).")

    p_chart = prod_df[prod_df["Region"] != "TOTAL"].copy()

    col_p1, col_p2 = st.columns([1, 1.3])

    with col_p1:
        prod_totals = {
            "OS (Owner Serviced)": p_chart["OS (Owner Serviced)"].sum(),
            "IPRE": p_chart["IPRE"].sum(),
            "Wealth": p_chart["Wealth"].sum(),
            "Islamic": p_chart["Islamic"].sum(),
            "PSB (Public Sector)": p_chart["PSB (Public Sector)"].sum(),
        }
        fig_donut = px.pie(
            names=list(prod_totals.keys()),
            values=list(prod_totals.values()),
            hole=0.48,
            title=f"<b>Product Contribution Share ({horizon})</b>",
            color=list(prod_totals.keys()),
            color_discrete_map=PRODUCT_COLOR_MAP
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Segoe UI, sans-serif", color="#334155"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=390
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_p2:
        fig_prod_bar = go.Figure()
        for p_col, p_color in [
            ("OS (Owner Serviced)", PRODUCT_COLOR_MAP["OS (Owner Serviced)"]),
            ("IPRE", PRODUCT_COLOR_MAP["IPRE"]),
            ("Wealth", PRODUCT_COLOR_MAP["Wealth"]),
            ("Islamic", PRODUCT_COLOR_MAP["Islamic"]),
            ("PSB (Public Sector)", PRODUCT_COLOR_MAP["PSB (Public Sector)"]),
        ]:
            fig_prod_bar.add_trace(go.Bar(x=p_chart["Region"], y=p_chart[p_col], name=p_col, marker_color=p_color))

        fig_prod_bar.update_layout(
            title=f"<b>Regional Payouts by Product Group</b>",
            barmode="stack",
            xaxis_tickangle=-40
        )
        apply_pbi_chart_layout(fig_prod_bar, height=390)
        st.plotly_chart(fig_prod_bar, use_container_width=True)

    formatted_prod = prod_df.copy()
    for col in ["OS (Owner Serviced)", "IPRE", "Wealth", "Islamic", "PSB (Public Sector)", "Total Payout", f"{horizon} Target"]:
        formatted_prod[col] = formatted_prod[col].apply(lambda x: f"R {x:,.0f}")
    formatted_prod["% Achieved"] = formatted_prod["% Achieved"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(formatted_prod, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 3: Dealmaker Ranking & League Table
# ---------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-title">Dealmaker League Table & Performance Movements</div>', unsafe_allow_html=True)
    st.caption("Replication of 'DM Ranking Preggie' and 'Dealmaker Ranking' sheets.")

    if dm_col and dm_col in raw_df.columns:
        if payout_col and payout_col in raw_df.columns:
            core_df = raw_df[raw_df[payout_col].astype(str).str.strip().str.upper().isin(CORE_PAYOUT_TYPES)]
        else:
            core_df = raw_df

        if week_col and week_col in core_df.columns and pd.to_numeric(core_df[week_col], errors="coerce").notnull().any():
            core_df[week_col] = pd.to_numeric(core_df[week_col], errors="coerce")
            max_wk = int(core_df[week_col].max())
            prev_wk = max(1, max_wk - 1)

            dm_latest = core_df[core_df[week_col] == max_wk].groupby(dm_col)[amount_col].sum().rename("Latest Payout")
            dm_prev = core_df[core_df[week_col] == prev_wk].groupby(dm_col)[amount_col].sum().rename("Prev Payout")

            dm_table = pd.concat([dm_latest, dm_prev], axis=1).fillna(0.0).reset_index()
            dm_table["Weekly Budget"] = 8_500_000.0
            dm_table["Latest % Achieved"] = (dm_table["Latest Payout"] / dm_table["Weekly Budget"]) * 100.0
            dm_table["Prev % Achieved"] = (dm_table["Prev Payout"] / dm_table["Weekly Budget"]) * 100.0

            dm_table["Latest Rank"] = dm_table["Latest Payout"].rank(ascending=False, method="min").astype(int)
            dm_table["Prev Rank"] = dm_table["Prev Payout"].rank(ascending=False, method="min").astype(int)
            dm_table["Rank Movement"] = dm_table["Prev Rank"] - dm_table["Latest Rank"]
            dm_table.sort_values(by="Latest Rank", inplace=True)

            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                # Top 10 horizontal bar with corporate teal
                top10_dm = dm_table.head(10).sort_values("Latest Payout", ascending=True)
                fig_dm = px.bar(
                    top10_dm,
                    x="Latest Payout",
                    y=dm_col,
                    orientation="h",
                    title=f"<b>Top 10 Dealmakers — Latest Week (Week {max_wk})</b>",
                    color_discrete_sequence=[COLOR_PRIMARY_TEAL]
                )
                apply_pbi_chart_layout(fig_dm, height=430)
                st.plotly_chart(fig_dm, use_container_width=True)

            with col_r2:
                st.write("##### 🏅 Podium Leaders")
                for _, r in dm_table.head(3).iterrows():
                    move_str = f"▲ +{abs(r['Rank Movement'])}" if r["Rank Movement"] > 0 else (f"▼ -{abs(r['Rank Movement'])}" if r["Rank Movement"] < 0 else "— 0")
                    badge_col = "#059669" if r["Rank Movement"] >= 0 else "#DC2626"
                    st.markdown(f"""
                    <div style="padding:12px; margin-bottom:10px; background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {COLOR_PRIMARY_TEAL}; border-radius:6px;">
                        <span style="font-size:0.8rem; font-weight:700; color:#64748B;">RANK #{r['Latest Rank']}</span><br>
                        <b style="color:#002D38; font-size:1.05rem;">{r[dm_col]}</b><br>
                        Payout: <b>R {r['Latest Payout']:,.0f}</b> ({r['Latest % Achieved']:.1f}%)<br>
                        <span style="color:{badge_col}; font-weight:700; font-size:0.8rem;">Shift: {move_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

            disp_dm = dm_table.copy()
            for col in ["Latest Payout", "Prev Payout", "Weekly Budget"]:
                disp_dm[col] = disp_dm[col].apply(lambda x: f"R {x:,.0f}")
            disp_dm["Latest % Achieved"] = disp_dm["Latest % Achieved"].apply(lambda x: f"{x:.1f}%")
            disp_dm["Prev % Achieved"] = disp_dm["Prev % Achieved"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(disp_dm, use_container_width=True, hide_index=True)
        else:
            dm_agg = core_df.groupby(dm_col)[amount_col].sum().reset_index()
            dm_agg.columns = [dm_col, "Total Payout"]
            dm_agg["Rank"] = dm_agg["Total Payout"].rank(ascending=False, method="min").astype(int)
            dm_agg.sort_values(by="Rank", inplace=True)
            dm_agg["Total Payout"] = dm_agg["Total Payout"].apply(lambda x: f"R {x:,.0f}")
            st.dataframe(dm_agg, use_container_width=True, hide_index=True)
    else:
        st.info("No Dealmaker column (DM_NAME) detected in current dataset.")


# ---------------------------------------------------------
# TAB 4: Weekly Payout Tracker & Trends
# ---------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-title">Weekly Tracker & Trend Analytics</div>', unsafe_allow_html=True)
    st.caption("Replication of 'Weekly Tracker' and 'PAYOUT VS BUDGET' historical curve.")

    if week_col and week_col in raw_df.columns:
        if payout_col and payout_col in raw_df.columns:
            core_weekly = raw_df[raw_df[payout_col].astype(str).str.strip().str.upper().isin(CORE_PAYOUT_TYPES)].copy()
        else:
            core_weekly = raw_df.copy()

        core_weekly[week_col] = pd.to_numeric(core_weekly[week_col], errors="coerce")
        core_weekly = core_weekly[core_weekly[week_col].notnull()]

        # 1. Total Aggregated Trend vs Target Line
        weekly_totals = core_weekly.groupby(week_col)[amount_col].sum().reset_index()
        weekly_totals["Weekly Target"] = sum(DEFAULT_WEEKLY_TARGETS.values())

        fig_agg = go.Figure()
        fig_agg.add_trace(go.Scatter(
            x=weekly_totals[week_col],
            y=weekly_totals[amount_col],
            mode="lines+markers",
            name="Actual Payouts",
            line=dict(color=COLOR_PRIMARY_TEAL, width=3.5),
            marker=dict(size=7, color=COLOR_NAVY_DARK)
        ))
        fig_agg.add_trace(go.Scatter(
            x=weekly_totals[week_col],
            y=weekly_totals["Weekly Target"],
            mode="lines",
            name="Weekly Target (Consolidated)",
            line=dict(color=COLOR_TARGET_LINE, width=2.5, dash="dash")
        ))
        fig_agg.update_layout(
            title="<b>Consolidated Weekly CPF Payouts vs Budget Benchmark</b>",
            xaxis_title="Fiscal Week",
            yaxis_title="Total Payouts (ZAR)"
        )
        apply_pbi_chart_layout(fig_agg, height=360)
        st.plotly_chart(fig_agg, use_container_width=True)

        # 2. Regional Multi-Line Progression
        if region_col and region_col in core_weekly.columns:
            trend_df = core_weekly.groupby([week_col, region_col])[amount_col].sum().reset_index()
            fig_trend = px.line(
                trend_df,
                x=week_col,
                y=amount_col,
                color=region_col,
                markers=True,
                title="<b>Regional Weekly Progression Trend</b>",
                labels={week_col: "Fiscal Week", amount_col: "Payout Amount (ZAR)"},
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            apply_pbi_chart_layout(fig_trend, height=440)
            fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5))
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Weekly trend requires historical records with a WEEK column.")

st.markdown("---")
st.markdown('<div style="text-align:center; color:#94A3B8; font-size:0.8rem;">CPF & CAS Commercial Banking Dashboard • Power BI UI Engine</div>', unsafe_allow_html=True)

"""
app.py - CPF & CAS Payouts Dashboard
Replication of Power BI & Excel Analysis Workbook
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
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CPF & CAS Payouts Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .positive { color: #16a34a; }
    .negative { color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
@st.cache_data
def load_excel_file(file, sheet_name=None) -> tuple[pd.DataFrame, list]:
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
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=56)
st.sidebar.title("CPF & CAS Dashboard")
st.sidebar.caption("Commercial Property Finance Payouts Engine")

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
        selected_sheet = st.sidebar.selectbox("Select Excel Sheet", xl_probe.sheet_names)
        raw_df, _ = load_excel_file(uploaded_file, selected_sheet)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    raw_df = generate_mock_data()
    st.sidebar.info("Using Benchmark Dataset (Weeks 1 to 24)")

# Horizon Slicer (WTD vs MTD vs YTD)
st.sidebar.markdown("---")
st.sidebar.subheader("Horizon View")
horizon = st.sidebar.radio(
    "Select Performance Window:",
    options=["WTD", "MTD", "YTD"],
    index=0,
    help="WTD = Latest Max Week | MTD = Latest Month | YTD = Full Fiscal Period"
)

# Super Region Slicer
st.sidebar.markdown("---")
all_super_regions = ["ALL"] + sorted(list(set(REGION_SUPER_REGION_MAP.values())))
selected_super = st.sidebar.selectbox("Super Region Filter", all_super_regions)

# ---------------------------------------------------------
# Column Resolution & Filtering
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

# Filter by horizon
filtered_horizon_df = filter_by_horizon(raw_df, horizon)

# Filter by Super Region
if selected_super != "ALL" and region_col and region_col in filtered_horizon_df.columns:
    allowed_regions = [r for r, s in REGION_SUPER_REGION_MAP.items() if s == selected_super]
    filtered_horizon_df = filtered_horizon_df[
        filtered_horizon_df[region_col].astype(str).str.strip().str.upper().isin(allowed_regions)
    ]

# Calculate Sales Indicators & Product Analysis
sales_df = calculate_sales_indicators(filtered_horizon_df, horizon)
prod_df = calculate_product_group_analysis(filtered_horizon_df, horizon)

# Totals
total_sales_row = sales_df[sales_df["Region"] == "TOTAL"].iloc[0]
total_payout = total_sales_row["Total Paid Out"]
total_target = total_sales_row[f"{horizon} Target"]
achieve_pct = total_sales_row["% Target Achieved"]
growth_total = total_sales_row["Growth Payouts"]
restr_total = total_sales_row["Restructures"]
readv_total = total_sales_row["Re-advance Payouts"]

# ---------------------------------------------------------
# Top KPI Cards
# ---------------------------------------------------------
st.title("Commercial Property Finance & CAS Payouts")
st.markdown(f"**Execution View:** `{horizon}` | **Super Region:** `{selected_super}` | **Rule Definition:** `ANALYSIS.xlsx` Specification")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{horizon} Total Payouts</div>
        <div class="metric-val">R {total_payout:,.0f}</div>
        <div class="metric-delta {'positive' if achieve_pct >= 100 else 'negative'}">{achieve_pct:.1f}% of Target</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{horizon} Target</div>
        <div class="metric-val">R {total_target:,.0f}</div>
        <div class="metric-delta">Target Multiplier: x{get_target_multiplier(horizon):.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Growth Payouts</div>
        <div class="metric-val">R {growth_total:,.0f}</div>
        <div class="metric-delta">{(growth_total/total_payout*100 if total_payout>0 else 0):.1f}% of Payouts</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Restructures</div>
        <div class="metric-val">R {restr_total:,.0f}</div>
        <div class="metric-delta">{(restr_total/total_payout*100 if total_payout>0 else 0):.1f}% of Payouts</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Re-Advances</div>
        <div class="metric-val">R {readv_total:,.0f}</div>
        <div class="metric-delta">{(readv_total/total_payout*100 if total_payout>0 else 0):.1f}% of Payouts</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Regional Sales Indicators",
    "🏢 Product Group (CAS) Analysis",
    "🏆 Dealmaker Ranking & Movements",
    "📈 Weekly Payout Tracker & Trends",
])

# ---------------------------------------------------------
# TAB 1: Regional Sales Indicators
# ---------------------------------------------------------
with tabs[0]:
    st.subheader(f"Sales Indicators Consolidation - {horizon}")
    st.caption("Consolidated Growth, Restructures, Re-advances and Other Segment payouts vs regional targets.")

    chart_data = sales_df[sales_df["Region"] != "TOTAL"].copy()

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=chart_data["Region"],
        y=chart_data["Total Paid Out"],
        name="Actual Payouts",
        marker_color="#1e40af"
    ))
    fig_bar.add_trace(go.Bar(
        x=chart_data["Region"],
        y=chart_data[f"{horizon} Target"],
        name=f"{horizon} Target",
        marker_color="#94a3b8"
    ))
    fig_bar.update_layout(
        title=f"Total Paid Out vs {horizon} Target per Region",
        barmode="group",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=20, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Growth Payouts"], name="Growth", marker_color="#10b981"))
    fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Restructures"], name="Restructures", marker_color="#f59e0b"))
    fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Re-advance Payouts"], name="Re-advances", marker_color="#6366f1"))
    fig_stack.add_trace(go.Bar(x=chart_data["Region"], y=chart_data["Other Segment"], name="Other Segment", marker_color="#0ea5e9"))
    fig_stack.update_layout(
        title=f"Deal Composition Breakdown per Region ({horizon})",
        barmode="stack",
        xaxis_tickangle=-45,
        height=380,
        margin=dict(l=20, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    formatted_sales = sales_df.copy()
    for col in ["Growth Payouts", "Restructures", "Re-advance Payouts", "Other Segment", "Total Paid Out", f"{horizon} Target"]:
        formatted_sales[col] = formatted_sales[col].apply(lambda x: f"R {x:,.0f}")
    formatted_sales["% Target Achieved"] = formatted_sales["% Target Achieved"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(formatted_sales, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 2: Product Group (CAS) Analysis
# ---------------------------------------------------------
with tabs[1]:
    st.subheader(f"Product Group (CAS) Breakdown - {horizon}")
    st.caption("Breakdown across Owner Serviced (OS), IPRE, Wealth, Islamic (Shariah), and PSB (Public Sector).")

    p_chart = prod_df[prod_df["Region"] != "TOTAL"].copy()

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        prod_totals = {
            "OS": p_chart["OS (Owner Serviced)"].sum(),
            "IPRE": p_chart["IPRE"].sum(),
            "Wealth": p_chart["Wealth"].sum(),
            "Islamic": p_chart["Islamic"].sum(),
            "PSB": p_chart["PSB (Public Sector)"].sum(),
        }
        fig_donut = px.pie(
            names=list(prod_totals.keys()),
            values=list(prod_totals.values()),
            hole=0.45,
            title=f"Product Contribution ({horizon})",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_p2:
        fig_prod_bar = go.Figure()
        for p_col, p_color in [
            ("OS (Owner Serviced)", "#3b82f6"),
            ("IPRE", "#10b981"),
            ("Wealth", "#8b5cf6"),
            ("Islamic", "#f97316"),
            ("PSB (Public Sector)", "#ec4899"),
        ]:
            fig_prod_bar.add_trace(go.Bar(x=p_chart["Region"], y=p_chart[p_col], name=p_col, marker_color=p_color))
        fig_prod_bar.update_layout(
            title=f"Regional Split by Product ({horizon})",
            barmode="stack",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(l=20, r=20, t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prod_bar, use_container_width=True)

    formatted_prod = prod_df.copy()
    for col in ["OS (Owner Serviced)", "IPRE", "Wealth", "Islamic", "PSB (Public Sector)", "Total Payout", f"{horizon} Target"]:
        formatted_prod[col] = formatted_prod[col].apply(lambda x: f"R {x:,.0f}")
    formatted_prod["% Achieved"] = formatted_prod["% Achieved"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(formatted_prod, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 3: Dealmaker Ranking
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Dealmaker Performance & League Table")
    st.caption("Replication of 'DM Ranking Preggie' and 'Dealmaker Ranking' tabs from workbook.")

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
                fig_dm = px.bar(
                    dm_table.head(10),
                    x="Latest Payout",
                    y=dm_col,
                    orientation="h",
                    title=f"Top 10 Dealmakers - Latest Week (Week {max_wk})",
                    color="Latest % Achieved",
                    color_continuous_scale="Viridis"
                )
                fig_dm.update_layout(yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_dm, use_container_width=True)

            with col_r2:
                st.write("##### Top Performers")
                for _, r in dm_table.head(3).iterrows():
                    move_icon = "🟢 ▲" if r["Rank Movement"] > 0 else ("🔴 ▼" if r["Rank Movement"] < 0 else "⚪ —")
                    st.markdown(f"""
                    <div style="padding:10px; margin-bottom:8px; background:#f1f5f9; border-radius:6px;">
                        <b>#{r['Latest Rank']} {r[dm_col]}</b><br>
                        Payout: <b>R {r['Latest Payout']:,.0f}</b> ({r['Latest % Achieved']:.1f}%)<br>
                        <small>Movement: {move_icon} {abs(r['Rank Movement'])} positions</small>
                    </div>
                    """, unsafe_allow_html=True)

            disp_dm = dm_table.copy
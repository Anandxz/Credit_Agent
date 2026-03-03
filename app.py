"""
Credit Intelligence - Streamlit Frontend
Upload financial documents and extract key financial terms.
"""

import streamlit as st
import json
import tempfile
import os
from pathlib import Path

# Import agent
from document_intelligence_agent import run_agent_on_file, run_agent_on_text

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0e1a;
        color: #e0e6f0;
    }

    .main { background-color: #0a0e1a; padding: 2rem; }

    .header-block {
        border-left: 4px solid #00d4ff;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, rgba(0,212,255,0.07) 0%, transparent 100%);
    }

    .header-block h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #00d4ff;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .header-block p {
        color: #7a8aaa;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .card {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .card-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #00d4ff;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1e2a3a;
        padding-bottom: 0.5rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1a2332;
    }

    .metric-row:last-child { border-bottom: none; }

    .metric-label {
        color: #7a8aaa;
        font-size: 0.85rem;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #e0e6f0;
        font-weight: 600;
    }

    .metric-value.highlight { color: #00d4ff; }
    .metric-value.danger { color: #ff4757; }
    .metric-value.warn { color: #ffa502; }
    .metric-value.safe { color: #2ed573; }

    .risk-chip {
        display: inline-block;
        background: rgba(255,71,87,0.15);
        border: 1px solid rgba(255,71,87,0.4);
        color: #ff6b7a;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0.2rem;
    }

    .confidence-bar-bg {
        background: #1e2a3a;
        border-radius: 4px;
        height: 8px;
        margin-top: 0.3rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #0a0e1a;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #33ddff, #00bbee);
        transform: translateY(-1px);
    }

    .stFileUploader {
        background: #111827;
        border: 2px dashed #1e2a3a;
        border-radius: 8px;
        padding: 1rem;
    }

    .stTextArea > div > div > textarea {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 6px;
        color: #e0e6f0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 8px;
        padding: 0.3rem;
        gap: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #7a8aaa;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        border-radius: 6px;
        padding: 0.4rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #1e2a3a !important;
        color: #00d4ff !important;
    }

    .status-processing {
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #00d4ff;
        text-align: center;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .stSelectbox > div > div {
        background: #111827;
        border: 1px solid #1e2a3a;
        color: #e0e6f0;
    }

    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 8px;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #1e2a3a;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def format_inr(value):
    if value is None:
        return "—"
    try:
        v = float(value)
        if v >= 1e7:
            return f"Rs. {v/1e7:.2f} Cr"
        elif v >= 1e5:
            return f"Rs. {v/1e5:.2f} L"
        else:
            return f"Rs. {v:,.0f}"
    except Exception:
        return str(value)

def metric_row(label, value, cls=""):
    return f"""
    <div class="metric-row">
        <span class="metric-label">{label}</span>
        <span class="metric-value {cls}">{value}</span>
    </div>"""

def confidence_color(score):
    if score >= 80:
        return "safe"
    elif score >= 60:
        return "warn"
    else:
        return "danger"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="header-block">
    <h1>CREDIT INTELLIGENCE</h1>
    <p>Document Analysis System &nbsp;|&nbsp; Indian Financial Context &nbsp;|&nbsp; Powered by Gemini</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────

col_input, col_output = st.columns([1, 2], gap="large")

with col_input:
    st.markdown('<div class="card-title">INPUT DOCUMENT</div>', unsafe_allow_html=True)

    input_mode = st.selectbox(
        "Input Mode",
        ["Upload File (PDF / Image / TXT)", "Paste Raw Text"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    raw_text = None

    if input_mode == "Upload File (PDF / Image / TXT)":
        uploaded_file = st.file_uploader(
            "Drop your document here",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            help="Supports annual reports, GST returns, bank statements, scanned PDFs"
        )
        if uploaded_file:
            st.markdown(f"""
            <div style="background:#0d1f2d; border:1px solid #00d4ff33; border-radius:6px; padding:0.7rem 1rem; margin-top:0.5rem;">
                <span style="color:#00d4ff; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;">FILE LOADED</span><br>
                <span style="color:#e0e6f0; font-size:0.85rem;">{uploaded_file.name}</span><br>
                <span style="color:#7a8aaa; font-size:0.78rem;">{uploaded_file.size / 1024:.1f} KB &nbsp;|&nbsp; {uploaded_file.type}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        raw_text = st.text_area(
            "Paste document text",
            height=280,
            placeholder="Paste content from annual report, GST return, bank statement...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_btn = st.button("ANALYZE DOCUMENT", use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#7a8aaa; font-size:0.78rem; line-height:1.7;">
        <b style="color:#00d4ff;">SUPPORTED FORMATS</b><br>
        PDF Annual Reports &nbsp;|&nbsp; Scanned Docs<br>
        GST Returns (GSTR-1, 3B, 2A)<br>
        Bank Statements &nbsp;|&nbsp; Legal Notices<br>
        Sanction Letters &nbsp;|&nbsp; Audit Reports<br><br>
        <b style="color:#00d4ff;">DETECTION</b><br>
        Circular Trading &nbsp;|&nbsp; Revenue Inflation<br>
        GST Mismatch &nbsp;|&nbsp; Going Concern<br>
        High Leverage &nbsp;|&nbsp; Auditor Remarks
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYSIS & OUTPUT
# ─────────────────────────────────────────────

with col_output:

    if analyze_btn:
        if not uploaded_file and not raw_text:
            st.warning("Please upload a file or paste document text first.")
        else:
            with st.spinner(""):
                st.markdown('<div class="status-processing">PROCESSING DOCUMENT...</div>', unsafe_allow_html=True)

                result = None
                error = None

                try:
                    if uploaded_file:
                        suffix = Path(uploaded_file.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        result = run_agent_on_file(tmp_path)
                        os.unlink(tmp_path)
                    else:
                        result = run_agent_on_text(raw_text)

                    if result.get("error"):
                        error = result["error"]
                    else:
                        st.session_state["result"] = result["extracted_data"]

                except Exception as e:
                    error = str(e)

                if error:
                    st.error(f"Analysis failed: {error}")

    # ── RENDER RESULTS ──
    if "result" in st.session_state:
        data = st.session_state["result"]
        fin = data.get("financials", {})
        gst = data.get("gst_data", {})
        bank = data.get("banking_obligations", {})
        legal = data.get("legal_risks", {})
        ratios = data.get("key_financial_ratios", {})
        risk_signals = data.get("risk_signals", [])
        circular = data.get("circular_trading_analysis", {})
        score = data.get("data_confidence_score", 0)

        # Company header
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.5rem;">
            <div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:#e0e6f0; font-weight:700;">
                    {data.get('company_name', 'Unknown Company')}
                </div>
                <div style="color:#7a8aaa; font-size:0.8rem; margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.08em;">
                    {data.get('document_type', '')} &nbsp;|&nbsp; FY {fin.get('financial_year', 'N/A')}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem; color:{('#2ed573' if score>=80 else '#ffa502' if score>=60 else '#ff4757')}; font-weight:700;">
                    {score}
                </div>
                <div style="color:#7a8aaa; font-size:0.7rem; text-transform:uppercase;">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk signals
        if risk_signals:
            chips = "".join([f'<span class="risk-chip">{r}</span>' for r in risk_signals])
            st.markdown(f"""
            <div class="card" style="border-color:#ff475733;">
                <div class="card-title" style="color:#ff6b7a;">RISK SIGNALS DETECTED</div>
                <div>{chips}</div>
            </div>
            """, unsafe_allow_html=True)

        tabs = st.tabs(["FINANCIALS", "GST DATA", "BANKING", "LEGAL", "RAW JSON"])

        # ── TAB 1: FINANCIALS ──
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                rows = (
                    metric_row("Total Revenue", format_inr(fin.get("total_revenue")), "highlight") +
                    metric_row("Net Profit", format_inr(fin.get("net_profit")),
                               "safe" if (fin.get("net_profit") or 0) > 0 else "danger") +
                    metric_row("EBITDA", format_inr(fin.get("ebitda"))) +
                    metric_row("Gross Profit", format_inr(fin.get("gross_profit"))) +
                    metric_row("Operating Profit", format_inr(fin.get("operating_profit"))) +
                    metric_row("Interest Expense", format_inr(fin.get("interest_expense")), "warn") +
                    metric_row("Depreciation", format_inr(fin.get("depreciation")))
                )
                st.markdown(f'<div class="card"><div class="card-title">P&L STATEMENT</div>{rows}</div>', unsafe_allow_html=True)

            with c2:
                rows2 = (
                    metric_row("Total Assets", format_inr(fin.get("total_assets"))) +
                    metric_row("Total Debt", format_inr(fin.get("total_debt")), "warn") +
                    metric_row("Equity", format_inr(fin.get("equity"))) +
                    metric_row("Current Assets", format_inr(fin.get("current_assets"))) +
                    metric_row("Current Liabilities", format_inr(fin.get("current_liabilities")))
                )
                st.markdown(f'<div class="card"><div class="card-title">BALANCE SHEET</div>{rows2}</div>', unsafe_allow_html=True)

                dte = ratios.get("debt_to_equity")
                cr = ratios.get("current_ratio")
                npm = ratios.get("net_profit_margin")
                ic = ratios.get("interest_coverage")

                ratio_rows = (
                    metric_row("Debt / Equity", f"{dte:.2f}x" if dte else "—",
                               "danger" if dte and dte > 2 else "safe" if dte else "") +
                    metric_row("Current Ratio", f"{cr:.2f}x" if cr else "—",
                               "safe" if cr and cr > 1.5 else "warn" if cr else "") +
                    metric_row("Net Profit Margin", f"{npm:.1f}%" if npm else "—",
                               "safe" if npm and npm > 5 else "warn" if npm else "") +
                    metric_row("Interest Coverage", f"{ic:.2f}x" if ic else "—",
                               "safe" if ic and ic > 2 else "danger" if ic else "")
                )
                st.markdown(f'<div class="card"><div class="card-title">KEY RATIOS</div>{ratio_rows}</div>', unsafe_allow_html=True)

        # ── TAB 2: GST ──
        with tabs[1]:
            mismatch = gst.get("gst_mismatch_flag")
            mismatch_html = ""
            if mismatch is True:
                mismatch_html = '<div style="background:rgba(255,71,87,0.1);border:1px solid rgba(255,71,87,0.4);border-radius:6px;padding:0.6rem 1rem;color:#ff6b7a;font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;margin-bottom:1rem;">GST MISMATCH DETECTED — GSTR-1 vs GSTR-3B discrepancy found</div>'
            elif circular.get("suspected"):
                mismatch_html = '<div style="background:rgba(255,71,87,0.1);border:1px solid rgba(255,71,87,0.4);border-radius:6px;padding:0.6rem 1rem;color:#ff6b7a;font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;margin-bottom:1rem;">CIRCULAR TRADING SUSPECTED — Review evidence below</div>'

            gst_rows = (
                metric_row("GST Number", gst.get("gst_number") or "—") +
                metric_row("GSTR-1 Turnover", format_inr(gst.get("gstr1_turnover")), "highlight") +
                metric_row("GSTR-3B Turnover", format_inr(gst.get("gstr3b_turnover"))) +
                metric_row("ITC Claimed", format_inr(gst.get("itc_claimed")), "warn") +
                metric_row("Mismatch Flag", "YES" if mismatch else "NO" if mismatch is False else "—",
                           "danger" if mismatch else "safe" if mismatch is False else "")
            )
            st.markdown(f'<div class="card">{mismatch_html}<div class="card-title">GST DETAILS</div>{gst_rows}</div>', unsafe_allow_html=True)

            if circular.get("evidence"):
                evid_items = "".join([f'<li style="color:#ffa502; margin:0.3rem 0;">{e}</li>' for e in circular["evidence"]])
                st.markdown(f'<div class="card" style="border-color:#ffa50233;"><div class="card-title" style="color:#ffa502;">CIRCULAR TRADING EVIDENCE</div><ul style="margin:0; padding-left:1.2rem;">{evid_items}</ul></div>', unsafe_allow_html=True)

        # ── TAB 3: BANKING ──
        with tabs[2]:
            bank_rows = (
                metric_row("Existing Loans", format_inr(bank.get("existing_loans")), "warn") +
                metric_row("Total Credits", format_inr(bank.get("total_credits")), "safe") +
                metric_row("Total Debits", format_inr(bank.get("total_debits"))) +
                metric_row("Avg Monthly Balance", format_inr(bank.get("average_monthly_balance"))) +
                metric_row("Bounced Cheques", str(bank.get("bounced_cheques")) if bank.get("bounced_cheques") is not None else "—",
                           "danger" if bank.get("bounced_cheques") else "") +
                metric_row("EMI Observed", format_inr(bank.get("emi_observed")))
            )
            st.markdown(f'<div class="card"><div class="card-title">BANKING ANALYSIS</div>{bank_rows}</div>', unsafe_allow_html=True)

            if bank.get("loan_types"):
                loans = ", ".join(bank["loan_types"])
                st.markdown(f'<div class="card"><div class="card-title">LOAN TYPES</div><div style="color:#e0e6f0; font-size:0.9rem;">{loans}</div></div>', unsafe_allow_html=True)

            if bank.get("collateral_offered"):
                coll = "<br>".join([f"• {c}" for c in bank["collateral_offered"]])
                st.markdown(f'<div class="card"><div class="card-title">COLLATERAL</div><div style="color:#e0e6f0; font-size:0.9rem; line-height:1.8;">{coll}</div></div>', unsafe_allow_html=True)

        # ── TAB 4: LEGAL ──
        with tabs[3]:
            legal_rows = (
                metric_row("Litigation Count", str(legal.get("ongoing_litigation_count") or "—"),
                           "danger" if legal.get("ongoing_litigation_count") else "") +
                metric_row("Tax Notices", str(len(legal.get("tax_notices", []))) or "—")
            )
            st.markdown(f'<div class="card"><div class="card-title">LEGAL OVERVIEW</div>{legal_rows}</div>', unsafe_allow_html=True)

            if legal.get("auditor_remarks"):
                remarks = "".join([f'<div style="background:#1a0d0d; border-left:3px solid #ff4757; padding:0.5rem 0.8rem; margin:0.4rem 0; font-size:0.85rem; border-radius:0 4px 4px 0;">{r}</div>' for r in legal["auditor_remarks"]])
                st.markdown(f'<div class="card" style="border-color:#ff475733;"><div class="card-title" style="color:#ff6b7a;">AUDITOR REMARKS</div>{remarks}</div>', unsafe_allow_html=True)

            if legal.get("notable_cases"):
                cases = "".join([f'<div style="padding:0.4rem 0; border-bottom:1px solid #1e2a3a; color:#e0e6f0; font-size:0.85rem;">• {c}</div>' for c in legal["notable_cases"]])
                st.markdown(f'<div class="card"><div class="card-title">NOTABLE CASES</div>{cases}</div>', unsafe_allow_html=True)

        # ── TAB 5: RAW JSON ──
        with tabs[4]:
            st.code(json.dumps(data, indent=2), language="json")

    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:400px; color:#2a3a50; text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏦</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; letter-spacing:0.1em;">AWAITING DOCUMENT</div>
            <div style="font-size:0.8rem; margin-top:0.5rem; color:#1e2a3a;">Upload a file or paste text to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
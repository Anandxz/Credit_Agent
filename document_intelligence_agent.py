"""
Document Intelligence Agent v2
- Supports PDF, PNG, JPG, TXT
- Uses new google-genai SDK (not deprecated google.generativeai)
- Detects circular trading, GST mismatch, revenue inflation
- LangGraph compatible
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
from typing import TypedDict, Optional
from pathlib import Path

from google import genai
from langgraph.graph import StateGraph, END


# CONFIG
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}


# STATE
class AgentState(TypedDict):
    file_path: Optional[str]
    document_text: Optional[str]
    extracted_data: Optional[dict]
    error: Optional[str]


# SYSTEM PROMPT
SYSTEM_PROMPT = """You are a senior corporate credit analyst at a large Indian bank.

Extract ALL structured financial, GST, banking, and legal risk data from the provided document.
The document may be a scanned PDF, annual report, GST return (GSTR-1/3B/2A), bank statement, sanction letter, or legal notice.
The document may contain Hindi/Marathi/Gujarati text mixed with English.
Numbers may use Indian formats: 1,00,000 = 1 Lakh | 1,00,00,000 = 1 Crore.

UNIT CONVERSION - always return raw integers in INR:
  Crore x 10000000 | Lakh x 100000 | Thousand x 1000

CIRCULAR TRADING DETECTION:
Flag "Circular trading suspected" if ANY of:
- Same party appears as both buyer AND seller
- Round-number transactions repeating every month
- GST turnover significantly exceeds bank credits
- ITC claims without matching inward supply evidence

REVENUE INFLATION DETECTION:
- GSTR-1 turnover > financial revenue by more than 10 percent means "Revenue inflation suspected"
- Bank credits much less than reported revenue means "Revenue-banking mismatch"

GST RULES:
- GSTR-1 not equal to GSTR-3B means gst_mismatch_flag: true
- Missing GST data means null values, do NOT estimate

BANKING RULES:
- Extract EMIs, loan repayments, cheque bounce mentions
- High revenue + low bank inflow means risk signal

LEGAL RULES:
Detect: "Subject to", "Material uncertainty", "Going concern", "Emphasis of matter"
Add to auditor_remarks and risk_signals

ANTI-HALLUCINATION:
- Only extract EXPLICITLY stated values
- If not present return null
- Do NOT guess or estimate

RISK SIGNALS (short phrases only):
High leverage | Revenue inflation suspected | Circular trading suspected | GST mismatch |
Going concern doubt | Auditor qualification | Litigation risk | Cheque bounce |
Declining cash flow | Director change | Large contingent liabilities | Revenue-banking mismatch

CONFIDENCE SCORE:
Clear structured tables: 85-95 | Minor OCR noise: 60-80 | Heavy OCR damage: 40-60 | Missing critical fields: less than 40

Return STRICT JSON only. No explanation. No markdown. No extra text.

{
  "company_name": "",
  "document_type": "",
  "financials": {
    "financial_year": "",
    "total_revenue": null,
    "net_profit": null,
    "ebitda": null,
    "total_debt": null,
    "equity": null,
    "current_assets": null,
    "current_liabilities": null,
    "gross_profit": null,
    "operating_profit": null,
    "interest_expense": null,
    "depreciation": null,
    "total_assets": null
  },
  "gst_data": {
    "gstr1_turnover": null,
    "gstr3b_turnover": null,
    "itc_claimed": null,
    "gst_mismatch_flag": null,
    "gst_number": ""
  },
  "banking_obligations": {
    "existing_loans": null,
    "loan_types": [],
    "collateral_offered": [],
    "average_monthly_balance": null,
    "total_credits": null,
    "total_debits": null,
    "bounced_cheques": null,
    "emi_observed": null
  },
  "legal_risks": {
    "ongoing_litigation_count": null,
    "notable_cases": [],
    "auditor_remarks": [],
    "tax_notices": []
  },
  "circular_trading_analysis": {
    "suspected": false,
    "evidence": []
  },
  "key_financial_ratios": {
    "debt_to_equity": null,
    "current_ratio": null,
    "net_profit_margin": null,
    "interest_coverage": null
  },
  "risk_signals": [],
  "data_confidence_score": 0,
  "ocr_quality": ""
}"""


# NODE 1: LOAD FILE
def load_document(state: AgentState) -> AgentState:
    file_path = state.get("file_path")
    document_text = state.get("document_text")

    if document_text and not file_path:
        return state

    if not file_path:
        return {**state, "error": "No file path or document text provided."}

    path = Path(file_path)
    if not path.exists():
        return {**state, "error": f"File not found: {file_path}"}

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return {**state, "error": f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}"}

    if suffix == ".txt":
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {**state, "document_text": text}
        except Exception as e:
            return {**state, "error": f"Failed to read text file: {e}"}

    return state


# NODE 2: EXTRACT WITH GEMINI
def extract_document_data(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    file_path = state.get("file_path")
    document_text = state.get("document_text")
    raw = ""

    try:
        if document_text and not file_path:
            # Plain text input
            full_prompt = SYSTEM_PROMPT + "\n\nExtract all information from this financial document:\n\n" + document_text
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )

        elif file_path:
            path = Path(file_path)
            suffix = path.suffix.lower()
            mime_map = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg"
            }
            mime_type = mime_map.get(suffix, "application/octet-stream")

            # Upload file path directly to Gemini File API
            uploaded_file = client.files.upload(
                file=Path(file_path),
                config={"mime_type": mime_type, "display_name": Path(file_path).name}
            )

            file_prompt = (
                SYSTEM_PROMPT +
                "\n\nThis is a financial document from an Indian company. "
                "It may be scanned or OCR-extracted with noise. "
                "Extract all financial, GST, banking, and legal information carefully. "
                "Pay special attention to Indian number formats (Lakh, Crore). "
                "Return strict JSON only."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, file_prompt],
            )
        else:
            return {**state, "error": "No input document provided."}

        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        return {**state, "extracted_data": parsed, "error": None}

    except json.JSONDecodeError as e:
        return {**state, "error": f"JSON parse error: {e}. Raw output: {raw[:400]}"}
    except Exception as e:
        return {**state, "error": str(e)}


# NODE 3: POST-PROCESS
def post_process(state: AgentState) -> AgentState:
    if state.get("error") or not state.get("extracted_data"):
        return state

    data = state["extracted_data"]
    fin = data.get("financials", {})
    gst = data.get("gst_data", {})
    bank = data.get("banking_obligations", {})
    ratios = data.get("key_financial_ratios", {})
    risk_signals = data.get("risk_signals", [])

    # Compute missing ratios
    try:
        debt = fin.get("total_debt")
        equity = fin.get("equity")
        if debt and equity and equity != 0 and not ratios.get("debt_to_equity"):
            ratios["debt_to_equity"] = round(debt / equity, 2)

        ca = fin.get("current_assets")
        cl = fin.get("current_liabilities")
        if ca and cl and cl != 0 and not ratios.get("current_ratio"):
            ratios["current_ratio"] = round(ca / cl, 2)

        revenue = fin.get("total_revenue")
        net_profit = fin.get("net_profit")
        if revenue and net_profit and revenue != 0 and not ratios.get("net_profit_margin"):
            ratios["net_profit_margin"] = round((net_profit / revenue) * 100, 2)

        ebit = fin.get("operating_profit") or fin.get("ebitda")
        interest = fin.get("interest_expense")
        if ebit and interest and interest != 0 and not ratios.get("interest_coverage"):
            ratios["interest_coverage"] = round(ebit / interest, 2)
    except Exception:
        pass

    # Cross-check GST vs bank credits
    try:
        gst1 = gst.get("gstr1_turnover")
        bank_credits = bank.get("total_credits")
        if gst1 and bank_credits:
            ratio = gst1 / bank_credits
            if ratio > 1.5 and "Revenue-banking mismatch" not in risk_signals:
                risk_signals.append("Revenue-banking mismatch")
            if ratio > 2.5:
                circ = data.get("circular_trading_analysis", {})
                circ["suspected"] = True
                evid = circ.get("evidence", [])
                msg = "GST turnover >> bank credits (ratio > 2.5x)"
                if msg not in evid:
                    evid.append(msg)
                circ["evidence"] = evid
                data["circular_trading_analysis"] = circ
                if "Circular trading suspected" not in risk_signals:
                    risk_signals.append("Circular trading suspected")
    except Exception:
        pass

    # High leverage check
    try:
        dte = ratios.get("debt_to_equity")
        if dte and dte > 2.0 and "High leverage" not in risk_signals:
            risk_signals.append("High leverage")
    except Exception:
        pass

    data["key_financial_ratios"] = ratios
    data["risk_signals"] = risk_signals
    state["extracted_data"] = data
    return state


# BUILD LANGGRAPH
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("load", load_document)
    graph.add_node("extract", extract_document_data)
    graph.add_node("postprocess", post_process)
    graph.set_entry_point("load")
    graph.add_edge("load", "extract")
    graph.add_edge("extract", "postprocess")
    graph.add_edge("postprocess", END)
    return graph.compile()


# PUBLIC API
def run_agent_on_file(file_path: str) -> dict:
    """Run agent on a PDF, image, or text file."""
    agent = build_agent()
    return agent.invoke({
        "file_path": file_path,
        "document_text": None,
        "extracted_data": None,
        "error": None
    })

def run_agent_on_text(text: str) -> dict:
    """Run agent on plain text input."""
    agent = build_agent()
    return agent.invoke({
        "file_path": None,
        "document_text": text,
        "extracted_data": None,
        "error": None
    })
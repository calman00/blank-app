# Calyx Cure ROI Calculator (Playground‑safe)
# -----------------------------------------------------------
# Minimal‑input ROI with auto packaging costs (dynamic case tiers) and
# quality deltas from the independent Cannabis Research Coalition (CRC) cure study.
# Deps: Streamlit + Pandas only.

import math
import streamlit as st
import pandas as pd
from email.message import EmailMessage
from urllib.parse import quote

# -------- Constants (CRC‑based weight retention) --------
GRAMS_PER_LB = 453.59237
ADVANTAGE_DEFAULT = {
    "Grove Bags": 635.07,   # g per 100 lb (Calyx advantage)
    "Turkey Bags": 657.67,  # g per 100 lb (Calyx advantage)
}

# -------- CRC decarboxylation results (8‑week cure) --------
# Starting decarb and end‑point by packaging (from Shortened CRC Research Report)
DECARB_START_PCT = 2.50
DECARB_AFTER_PCT = {
    "Calyx": 6.00,
    "Grove Bags": 4.66,
    "Turkey Bags": 6.21,
}

# -------- CRC week‑8 potency & terpene data (percent, dry basis) --------
THC_TABLE = {
    "D9-THC": {"Start": 0.43733, "Grove Bags": 1.915, "Calyx": 2.509, "Turkey Bags": 2.209},
    "THCA":   {"Start": 16.86800, "Grove Bags": 14.431, "Calyx": 16.419, "Turkey Bags": 15.586},
}

TERPENE_TABLE = {
    # Totals
    "Total Terpenes": {"Start": 0.92833, "Grove Bags": 0.739, "Calyx": 0.762, "Turkey Bags": 0.637},
    # Key aroma terpenes
    "Limonene":            {"Start": 2.37100,  "Grove Bags": 1.579, "Calyx": 2.114, "Turkey Bags": 1.488},
    "Beta-Myrcene":        {"Start": 1.84933,  "Grove Bags": 1.169, "Calyx": 1.466, "Turkey Bags": 1.015},
    "Beta-Caryophyllene":  {"Start": 1.42867,  "Grove Bags": 1.196, "Calyx": 1.247, "Turkey Bags": 1.011},
    "Linalool":            {"Start": 0.81033,  "Grove Bags": 0.546, "Calyx": 0.629, "Turkey Bags": 0.505},
    "Alpha-Terpineol":     {"Start": 0.62167,  "Grove Bags": 0.464, "Calyx": 0.548, "Turkey Bags": 0.457},
    "Alpha Pinene":        {"Start": 0.03010,  "Grove Bags": 0.312, "Calyx": 0.398, "Turkey Bags": 0.274},
    "Beta Pinene":         {"Start": 0.04223,  "Grove Bags": 0.350, "Calyx": 0.466, "Turkey Bags": 0.331},
}

# -------- Case‑break pricing (per CASE of 100 bags) --------
# Calyx Cure (from your pricing image)
CALYX_CASE = {
    "1lb":  {1:349, 2:325, 5:315, 10:299, 25:289, 50:279, 100:275},
    "5lb":  {1:499, 2:475, 5:465, 10:449, 25:439, 50:429, 100:420},
    "10lb": {1:799, 2:775, 5:765, 10:749, 25:749, 50:729, 100:725},
}

# Grove Bags (Opaque Pouches, case pack = 100). From the case‑break PDF.
# MSRP (1 case) + discounts at 3,6,12,25,50 cases.
GROVE_CASE = {
    "1lb":  {1:399.00, 3:379.05, 6:359.10, 12:339.15, 25:319.20, 50:299.25},
    "5lb":  {1:549.00, 3:521.55, 6:494.10, 12:466.65, 25:439.20, 50:411.75},
    "10lb": {1:1049.00,3:996.55, 6:944.10, 12:891.65, 25:839.20, 50:786.75},
}

# -------- Helpers --------
def price_per_g(price_per_lb: float) -> float:
    return price_per_lb / GRAMS_PER_LB


def per_bag_from_case(table: dict, size_key: str, cases: int):
    """Return ($/bag, applied_tier_cases, $/case) using floor‑to‑bracket (100 bags/case)."""
    tiers = table[size_key]
    available = sorted(tiers.keys())
    c = max(int(cases), 1)
    applied = max([t for t in available if t <= c], default=available[0])
    case_price = tiers[applied]
    return case_price / 100.0, applied, case_price


def compute(
    batch_lb: float,
    price_per_lb: float,
    bag_size_label: str,          # "1 lb", "5 lb", "10 lb"
    calyx_cases: int,
    current_packaging: str,       # "Grove Bags" or "Turkey Bags"
    grove_cases: int,             # ignored when Turkey
    price_premium_pct: float = 0.0,
    shrink_reduction_pct: float = 0.0,
):
    size_key = {"1 lb":"1lb", "5 lb":"5lb", "10 lb":"10lb"}[bag_size_label]

    # Bag counts (capacity = bag size)
    bags = max(1, math.ceil(batch_lb / float(bag_size_label.split()[0])))

    # $/bag by brand (with applied tier info)
    calyx_per_bag, calyx_tier, calyx_case_price = per_bag_from_case(CALYX_CASE, size_key, calyx_cases)
    if current_packaging == "Grove Bags":
        current_per_bag, grove_tier, grove_case_price = per_bag_from_case(GROVE_CASE, size_key, grove_cases)
        current_tier = grove_tier
        current_case_price = grove_case_price
    else:  # Turkey
        current_per_bag = calyx_per_bag * 0.25
        current_tier = None
        current_case_price = None

    # Packaging Δ per batch
    pack_delta = bags * (calyx_per_bag - current_per_bag)

    # Weight‑based revenue (no conservatism multiplier)
    adv = ADVANTAGE_DEFAULT[current_packaging]
    delta_g = adv * (batch_lb / 100.0)
    ppg = price_per_g(price_per_lb)
    incr_rev_weight = delta_g * ppg

    # Optional monetization
    base_rev = batch_lb * GRAMS_PER_LB * ppg
    premium_rev = base_rev * (price_premium_pct / 100.0)
    shrink_save = base_rev * (shrink_reduction_pct / 100.0)

    gross_gain = incr_rev_weight + premium_rev + shrink_save
    net_gain = gross_gain - pack_delta

    roi = None
    payback = None
    if pack_delta > 0 and gross_gain > 0:
        roi = (net_gain / pack_delta) * 100.0
        payback = pack_delta / gross_gain

    return {
        "bags": bags,
        "calyx_per_bag": calyx_per_bag,
        "current_per_bag": current_per_bag,
        "pack_delta": pack_delta,
        "delta_g": delta_g,
        "incr_rev_weight": incr_rev_weight,
        "premium_rev": premium_rev,
        "shrink_save": shrink_save,
        "gross_gain": gross_gain,
        "net_gain": net_gain,
        "roi": roi,
        "payback": payback,
        "calyx_applied_tier": calyx_tier,
        "current_applied_tier": current_tier,
        "calyx_case_price": calyx_case_price,
        "current_case_price": current_case_price,
    }

# -------- UI --------
st.title("Calyx Cure ROI Calculator-")
st.caption("Pick your bag size and order tiers. We auto‑calculate per‑bag costs and ROI from the independent Cannabis Research Coalition Cure study.")

colA, colB = st.columns(2)
with colA:
    batch_lb = st.number_input("Batch size (lb)", min_value=1.0, value=100.0, step=1.0)
    price_per_lb = st.number_input("Flower price ($/lb)", min_value=100.0, value=800.0, step=25.0)
    bag_size_label = st.selectbox("Bag size", ["1 lb", "5 lb", "10 lb"], index=0)
with colB:
    calyx_cases = st.number_input("Your Calyx order (cases)", min_value=1, value=1, step=1)
    current_packaging = st.selectbox("Your current packaging", ["Grove Bags", "Turkey Bags"], index=0)
    if current_packaging == "Grove Bags":
        grove_cases = st.number_input("Your Grove order (cases)", min_value=1, value=1, step=1)
    else:
        grove_cases = 1

# Advanced optional monetization (collapsed)
with st.expander("Advanced (optional): quality monetization & shrink"):
    price_premium_pct = st.number_input("Price premium from quality (%)", min_value=0.0, value=0.0, step=0.25)
    shrink_reduction_pct = st.number_input("Rework/shrink reduction (%)", min_value=0.0, value=0.0, step=0.25)

# Compute
k = compute(
    batch_lb=batch_lb,
    price_per_lb=price_per_lb,
    bag_size_label=bag_size_label,
    calyx_cases=calyx_cases,
    current_packaging=current_packaging,
    grove_cases=grove_cases,
    price_premium_pct=price_premium_pct,
    shrink_reduction_pct=shrink_reduction_pct,
)

# -------- Outputs --------
K1, K2, K3 = st.columns(3)
K1.metric("Net gain (per batch)", f"${k['net_gain']:,.0f}")
K2.metric("ROI", (f"{k['roi']:,.0f}%" if k['roi'] is not None else "—"))
K3.metric("Payback (batches)", (f"{k['payback']:.2f}" if k['payback'] is not None else "—"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calyx $/bag", f"${k['calyx_per_bag']:,.2f}")
c2.metric("Current $/bag", f"${k['current_per_bag']:,.2f}")
c3.metric("Packaging Δ (batch)", f"${k['pack_delta']:,.0f}")
c4.metric("Bags needed", f"{k['bags']:,}")

# show applied pricing brackets
st.caption(
    f"Calyx pricing bracket: {k['calyx_applied_tier']}-case tier ($"
    f"{k['calyx_case_price']:,.2f}/case → ${k['calyx_per_bag']:,.2f}/bag)."
)
if current_packaging == "Grove Bags":
    st.caption(
        f"Grove pricing bracket: {k['current_applied_tier']}-case tier ($"
        f"{k['current_case_price']:,.2f}/case → ${k['current_per_bag']:,.2f}/bag)."
    )
else:
    st.caption("Turkey bags priced at 25% of Calyx $/bag (no tiers).")

# ---------- THC & terpene data (from CRC) — Always visible ----------
st.subheader("THC & Terpenes (CRC 8‑week data)")
curr_label = current_packaging

# --- Total THC using week‑8 data ---

def total_thc(pkg: str) -> float:
    return THC_TABLE["D9-THC"][pkg] + 0.877 * THC_TABLE["THCA"][pkg]

calyx_total_thc = total_thc("Calyx")
curr_total_thc = total_thc(curr_label)
delta_total_thc = calyx_total_thc - curr_total_thc

M1, M2, M3 = st.columns(3)
M1.metric("Total THC (8 wks) — Calyx", f"{calyx_total_thc:.3f}%")
M2.metric(f"Total THC (8 wks) — {curr_label}", f"{curr_total_thc:.3f}%")
M3.metric("Δ Total THC (pp)", f"{delta_total_thc:+.3f}")

# --- Total terpene retention vs start ---
start_tt = TERPENE_TABLE["Total Terpenes"]["Start"]
calyx_tt = TERPENE_TABLE["Total Terpenes"]["Calyx"]
curr_tt = TERPENE_TABLE["Total Terpenes"][curr_label]
ret_calyx = (calyx_tt / start_tt) * 100.0 if start_tt > 0 else None
ret_curr = (curr_tt / start_tt) * 100.0 if start_tt > 0 else None
ret_uplift_pp = (ret_calyx - ret_curr) if (ret_calyx is not None and ret_curr is not None) else None

M4, M5, M6 = st.columns(3)
M4.metric("Total terpenes — Calyx", f"{calyx_tt:.3f}%")
M5.metric(f"Total terpenes — {curr_label}", f"{curr_tt:.3f}%")
M6.metric("Δ Total terpenes (pp)", f"{(calyx_tt - curr_tt):+.3f}")

R1, R2, R3 = st.columns(3)
R1.metric("Retention vs start — Calyx", f"{ret_calyx:.1f}%")
R2.metric(f"Retention vs start — {curr_label}", f"{ret_curr:.1f}%")
R3.metric("Retention uplift (pp)", f"{ret_uplift_pp:+.1f}")

# --- Key terpene retention table ---
key_names = [
    "Limonene","Beta-Myrcene","Beta-Caryophyllene",
    "Linalool","Alpha-Terpineol","Alpha Pinene","Beta Pinene"
]
rows = []
for name in key_names:
    start = TERPENE_TABLE[name]["Start"]
    cal = TERPENE_TABLE[name]["Calyx"]
    cur = TERPENE_TABLE[name][curr_label]
    if start and start > 0:
        r_cal = (cal / start) * 100.0
        r_cur = (cur / start) * 100.0
        uplift = r_cal - r_cur
    else:
        r_cal = None; r_cur = None; uplift = None
    rows.append({
        "Terpene": name,
        "Start %": start,
        "Calyx %": cal,
        f"{curr_label} %": cur,
        "Retention Calyx %": r_cal,
        f"Retention {curr_label} %": r_cur,
        "Retention uplift (pp)": uplift,
    })
terp_df = pd.DataFrame(rows)
st.dataframe(
    terp_df.style.format({
        "Start %": "{:.3f}", "Calyx %": "{:.3f}", f"{curr_label} %": "{:.3f}",
        "Retention Calyx %": "{:.1f}", f"Retention {curr_label} %": "{:.1f}",
        "Retention uplift (pp)": "+{:.1f}",
    }),
    use_container_width=True,
)

st.markdown("---")

st.subheader("Per‑batch value breakdown")
breakdown = pd.DataFrame({
    "Component": ["Packaging Δ (cost)", "Revenue (weight)", "Premium (quality)", "Shrink savings", "Net gain"],
    "USD": [ -k["pack_delta"], k["incr_rev_weight"], k["premium_rev"], k["shrink_save"], k["net_gain"] ],
})
st.bar_chart(breakdown.set_index("Component"))

# Annual rollup
batches_per_month = st.number_input("Batches per month", min_value=1, value=1, step=1)
annual_net = max(0.0, k["net_gain"]) * batches_per_month * 12
_plural = "batch" if batches_per_month == 1 else "batches"
st.metric(f"Annual net impact (assumes {batches_per_month} {_plural}/month)", f"${annual_net:,.0f}")

# ---------- Downloadable report (HTML) — Refined ----------
# Format data for clean tables
terp_df_report = terp_df.copy()
for col in ["Start %", "Calyx %", f"{curr_label} %"]:
    terp_df_report[col] = terp_df_report[col].map(lambda x: None if x is None else round(x, 3))
for col in ["Retention Calyx %", f"Retention {curr_label} %", "Retention uplift (pp)"]:
    terp_df_report[col] = terp_df_report[col].map(lambda x: None if x is None else round(x, 1))
terp_table_html = terp_df_report.to_html(index=False, border=0, classes="table")

breakdown_report = breakdown.copy()
breakdown_report["USD"] = breakdown_report["USD"].round(0)
breakdown_html = breakdown_report.to_html(index=False, border=0, classes="table")

# Helper values
ppg = price_per_lb / GRAMS_PER_LB
net_class = "good" if (k["net_gain"] or 0) >= 0 else "bad"
roi_text = (f"{k['roi']:.0f}%" if k['roi'] is not None else "—")
roi_class = "good" if (k["roi"] is not None and k["roi"] >= 0) else ("muted" if k["roi"] is None else "bad")
payback_text = (f"{k['payback']:.2f}" if k['payback'] is not None else "—")
pack_class = "bad" if k["pack_delta"] > 0 else ("good" if k["pack_delta"] < 0 else "muted")

# Takeaways
_takeaways = []
_takeaways.append(f"Extra sellable weight: <b>{k['delta_g']:,.0f} g</b> × ${ppg:.2f}/g = <b>${k['incr_rev_weight']:,.0f}</b> per batch")
if k["pack_delta"] != 0:
    sign = "cost" if k["pack_delta"] > 0 else "savings"
    _takeaways.append(f"Packaging {sign}: <b>${abs(k['pack_delta']):,.0f}</b> per batch")
# Quality notes
_t_thc = f"Total THC {'+' if delta_total_thc>=0 else ''}{delta_total_thc:.3f} pp"
_t_terp = f"Total terpenes {'+' if (calyx_tt-curr_tt)>=0 else ''}{(calyx_tt-curr_tt):.3f} pp; retention uplift {'+' if ret_uplift_pp>=0 else ''}{ret_uplift_pp:.1f} pp"
_takeaways.append(_t_thc)
_takeaways.append(_t_terp)

takeaways_html = "".join([f"<li>{t}</li>" for t in _takeaways])

# Science highlights: compute top-3 terpene uplift list
_top = sorted(rows, key=lambda r: (r["Retention uplift (pp)"] or -9999), reverse=True)[:3]
science_items = []
for r in _top:
    label = r["Terpene"]
    uplift = r["Retention uplift (pp)"]
    rcal = r["Retention Calyx %"]
    rcur = r[f"Retention {curr_label} %"]
    bar_w = max(0, min(100, rcal or 0))
    bar_w2 = max(0, min(100, rcur or 0))
    science_items.append(
        f"<div class='science-item'><div class='science-label'>{label}: +{uplift:.1f} pp</div>"
        f"<div class='bars'><div class='bar a' style='width:{bar_w:.1f}%' title='Calyx {rcal:.1f}%'></div>"
        f"<div class='bar b' style='width:{bar_w2:.1f}%' title='{curr_label} {rcur:.1f}%'></div></div></div>"
    )
science_html = "".join(science_items)

report_title = "Calyx Cure ROI Calculator-"
report_caption = "Pick your bag size and order tiers. We auto‑calculate per‑bag costs and ROI from the independent Cannabis Research Coalition Cure study."

report_html = f"""
<html>
<head>
<meta charset='utf-8'>
<style>
  :root {{
    --bg:#ffffff; --ink:#0f172a; --muted:#64748b; --muted2:#475569;
    --good:#0ea5a5; --bad:#ef4444; --card:#f8fafc; --line:#e5e7eb;
    --accent:#16a34a; --accentLite:#ecfdf5; --accentInk:#065f46;
  }}
  * {{ box-sizing:border-box; }}
  body {{ font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial; color:var(--ink); background:var(--bg); margin:24px; }}
  h1 {{ font-size:28px; margin:0 0 4px 0; letter-spacing:.2px; }}
  .sub {{ color:var(--muted); margin-bottom:18px; }}
  .hero {{ background:linear-gradient(135deg, var(--accentLite), #fff); border:1px solid var(--line); border-radius:16px; padding:16px; margin-bottom:16px; }}
  .kpis {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; margin-top:8px; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; box-shadow:0 1px 0 rgba(0,0,0,.02); }}
  .kpi .label {{ font-size:12px; color:var(--muted); }}
  .kpi .value {{ font-size:22px; font-weight:700; }}
  .kpi.good .value {{ color:var(--accentInk); }}
  .kpi.bad .value {{ color:var(--bad); }}
  .grid2 {{ display:grid; grid-template-columns: 1.2fr 0.8fr; gap:16px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px; margin:12px 0; box-shadow:0 1px 0 rgba(0,0,0,.02); }}
  .card h2 {{ font-size:18px; margin:0 0 8px 0; letter-spacing:.2px; }}
  .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--line); background:var(--card); font-size:12px; color:var(--muted2); margin:4px 8px 0 0; }}
  .table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .table th, .table td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }}
  .table tr:nth-child(odd) td {{ background:#fcfdff; }}
  ul.takeaways {{ margin:8px 0 0 20px; }}
  .progress {{ height:10px; background:#f1f5f9; border-radius:6px; overflow:hidden; }}
  .progress > div {{ height:100%; background:var(--accent); }}
  .muted {{ color:var(--muted); }}
  .small {{ font-size:12px; color:var(--muted); }}

  /* Science highlight block */
  .science {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
  .science .note {{ background:var(--accentLite); border:1px solid #d1fae5; color:var(--accentInk); border-radius:12px; padding:12px; }}
  .science .note h3 {{ margin:0 0 6px 0; font-size:14px; }}
  .science .note p {{ margin:6px 0; font-size:13px; }}
  .science .note .fn {{ font-size:12px; color:var(--muted2); margin-top:6px; }}
  .science .science-item {{ margin:6px 0; }}
  .science .science-label {{ font-size:13px; margin-bottom:6px; }}
  .science .bars {{ display:flex; gap:6px; align-items:center; }}
  .science .bar {{ height:8px; background:#e2e8f0; border-radius:6px; flex:1; position:relative; }}
  .science .bar.a {{ background:linear-gradient(90deg, var(--accent), #93c5fd); }}
  .science .bar.b {{ background:#cbd5e1; }}
</style>
</head>
<body>
  <div class='hero'>
    <h1>{report_title}</h1>
    <div class='sub'>{report_caption}</div>
    <div class='kpis'>
      <div class='kpi {net_class}'>
        <div class='label'>Net gain per batch</div>
        <div class='value'>${k['net_gain']:,.0f}</div>
      </div>
      <div class='kpi {roi_class}'>
        <div class='label'>ROI</div>
        <div class='value'>{roi_text}</div>
      </div>
      <div class='kpi'>
        <div class='label'>Payback (batches)</div>
        <div class='value'>{payback_text}</div>
      </div>
    </div>
  </div>

  <div class='grid2'>
    <div class='card'>
      <h2>What’s driving the value</h2>
      <ul class='takeaways'>
        {takeaways_html}
      </ul>
      <div style='margin-top:10px' class='small'>Flower price: ${price_per_lb:,.0f}/lb (≈ ${ppg:.2f}/g) · Batch size: {batch_lb} lb · Bags needed: {k['bags']:,}</div>
    </div>
    <div class='card'>
      <h2>Packaging pricing</h2>
      <div class='pill'>Bag size: {bag_size_label}</div>
      <div class='pill'>Calyx: {calyx_cases} cases → {k['calyx_applied_tier']}-case tier</div>
      <div class='pill'>Current: {current_packaging}{(' — ' + str(grove_cases) + ' cases → ' + str(k['current_applied_tier']) + '-case tier') if current_packaging=='Grove Bags' else ''}</div>
      <div style='margin-top:8px'>Calyx: <b>${k['calyx_case_price']:,.2f}/case</b> → <b>${k['calyx_per_bag']:,.2f}/bag</b></div>
      <div>Current: <b>{('$' + format(k['current_case_price'], ',.2f') + '/case → $' + format(k['current_per_bag'], ',.2f') + '/bag') if current_packaging=='Grove Bags' else 'Turkey at 25% of Calyx $/bag'}</b></div>
      <div class='small {pack_class}' style='margin-top:6px'>Packaging impact this batch: ${abs(k['pack_delta']):,.0f} ({'savings' if k['pack_delta']<0 else 'cost'})</div>
    </div>
  </div>

  <div class='card'>
    <h2>Per‑batch value breakdown</h2>
    {breakdown_html}
  </div>

  <div class='card'>
    <h2>Quality impact (CRC 8‑week)</h2>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:10px'>
      <div><div class='small muted'>Total THC — Calyx</div><div style='font-weight:700'>{calyx_total_thc:.3f}%</div></div>
      <div><div class='small muted'>Total THC — {curr_label}</div><div style='font-weight:700'>{curr_total_thc:.3f}%</div></div>
      <div><div class='small muted'>Δ Total THC</div><div style='font-weight:700; color:{'#0ea5a5' if delta_total_thc>=0 else '#ef4444'}'>{delta_total_thc:+.3f} pp</div></div>
    </div>

    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:10px'>
      <div><div class='small muted'>Total terpenes — Calyx</div><div style='font-weight:700'>{calyx_tt:.3f}%</div></div>
      <div><div class='small muted'>Total terpenes — {curr_label}</div><div style='font-weight:700'>{curr_tt:.3f}%</div></div>
      <div><div class='small muted'>Δ Total terpenes</div><div style='font-weight:700; color:{'#0ea5a5' if (calyx_tt-curr_tt)>=0 else '#ef4444'}'>{(calyx_tt - curr_tt):+.3f} pp</div></div>
    </div>

    <div style='display:grid;grid-template-columns:1fr 1fr; gap:12px;'>
      <div>
        <div class='small muted'>Retention vs start — Calyx</div>
        <div class='progress'><div style='width:{ret_calyx:.1f}%;'></div></div>
        <div class='small'>{ret_calyx:.1f}% retained</div>
      </div>
      <div>
        <div class='small muted'>Retention vs start — {curr_label}</div>
        <div class='progress'><div style='width:{ret_curr:.1f}%; background:#cbd5e1;'></div></div>
        <div class='small'>{ret_curr:.1f}% retained</div>
      </div>
    </div>
    <div class='small' style='margin-top:6px'><b>Retention uplift:</b> {ret_uplift_pp:+.1f} pp</div>

    <div style='margin-top:12px'>{terp_table_html}</div>
  </div>

  <div class='card'>
    <h2>Calyx Cure science highlights</h2>
    <div class='science'>
      <div class='note'>
        <h3>1) More sellable weight</h3>
        <p>At the processing stage, Calyx gained <b>+362.9 g</b> per 100 lb while Grove and Turkey <i>lost</i> <b>−272.2 g</b> / <b>−294.8 g</b>. That’s a net advantage of <b>+635–658 g/100 lb</b> vs common baselines.</p>
        <div class='fn'>From CRC Cure study weight data.</div>
      </div>
      <div class='note'>
        <h3>2) Potency & aroma preserved</h3>
        <p>Week‑8 Total THC is higher with Calyx and terpene <i>retention vs start</i> is improved. Below shows the top terpene retention gains (Calyx vs your current packaging).</p>
        <div class='science-list'>
          {science_html}
        </div>
      </div>
    </div>
  </div>

  <div class='card'>
    <h2>Methodology & sources</h2>
    <p class='small'>
      • <b>Weight revenue:</b> Δg = Advantage(g/100 lb) × (Batch lb / 100); ΔRev = Δg × (${ppg:.2f}/g).<br/>
      • <b>Total THC:</b> THC_total = Δ9‑THC + 0.877 × THCA (week‑8 dataset).<br/>
      • <b>Terpene retention:</b> Retention% = (Week‑8 value ÷ Start) × 100; uplift = Calyx − Current.<br/>
      • <b>Packaging cost impact:</b> #bags × (Calyx $/bag − Current $/bag), with per‑bag from case‑break tiers.<br/>
      • <b>Turkey price:</b> assumed at 25% of Calyx per‑bag (user‑adjustable rule on request).
    </p>
    <p class='small muted'>Study: Independent Cannabis Research Coalition (CRC) Cure study, 8‑week results.</p>
  </div>

  <div class='card'>
    <h2>Inputs</h2>
    <div>Batch size: {batch_lb} lb · Flower price: ${price_per_lb:,.0f}/lb (≈ ${ppg:.2f}/g) · Bag size: {bag_size_label}</div>
    <div>Throughput assumption: {batches_per_month} {('batch' if batches_per_month == 1 else 'batches')}/month → Annual net: ${(max(0.0, k['net_gain']) * batches_per_month * 12):,.0f}</div>
  </div>

  <div class='small muted'>This is an estimate; adjust inputs to match your operation.</div>
</body>
</html>
"""

st.download_button(
    "Download Your Report (HTML)",
    data=report_html.encode("utf-8"),
    file_name="Calyx_Cure_ROI_Report.html",
    mime="text/html",
)
   

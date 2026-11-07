"""Site Sentinel dashboard - photo + video safety checks, progress and weekly reports.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import glob
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from site_sentinel import config
from site_sentinel.annotate import draw_zones
from site_sentinel.pipeline import inspect_image, model_path_for
from site_sentinel.progress import build_checklist, compute_progress, confirm
from site_sentinel.report import WeekSummary, build_pdf, build_texts
from site_sentinel.video import VIDEO_EXTENSIONS, analyze_video, mm_ss
from site_sentinel.zones import Zone

st.set_page_config(page_title="Site Sentinel", page_icon="🦺", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- i18n ----
EN = {
    "title": "Site Sentinel", "sub": "AI construction safety & progress tracking - photos and videos",
    "photo": "Photo check", "video": "Video check", "live": "Live", "zones": "Zones",
    "progress": "Progress", "report": "Report", "accuracy": "Accuracy",
    "upload": "Drop a site photo here", "sample": "Or pick a sample photo",
    "run": "Run safety check", "blur": "Blur faces (privacy)",
    "model": "AI model", "accurate": "Accurate (yolo11m)", "fastm": "Fast (yolo11n)",
    "score": "Safety score", "workers": "Workers", "helmets": "Helmets OK", "vests": "Vests OK",
    "issues": "What needs attention", "none": "No violations detected. Good work.",
    "zone_hint": "Anyone whose feet are inside a restricted zone is flagged.",
    "save_zone": "Add zone", "clear_zone": "Clear all zones",
    "confirm": "Confirm stage", "supervisor": "Supervisor name", "timeline": "Check history",
    "gen_report": "Generate weekly report", "estimate": "Estimate only - confirm with a site supervisor.",
    "site": "Site name", "orig": "Original", "annot": "AI result", "dets": "Detections",
    "advice": "Advice", "vupload": "Drop a site video here (MP4, MOV, AVI, MKV, WEBM)",
    "vrun": "Analyse video", "rate": "Frames checked per second", "maxlen": "Analyse at most (minutes)",
    "avg": "Average score", "lowest": "Lowest score", "safe": "Safe time", "incidents": "Incidents",
    "maxw": "Most workers seen", "ptime": "Processing time", "chart": "Safety score over time",
    "playfrom": "Play from", "dl_video": "Download annotated video", "dl_csv": "Timeline (CSV)",
    "dl_json": "Full report (JSON)", "noinc": "No incidents - every checked moment was safe.",
    "vtip": "Tip: short clips (under 2 minutes) finish fastest on a laptop CPU.",
    "today": "Checks this session", "avg_all": "Average score", "inc_all": "Video incidents",
}
UR = {
    "title": "سائٹ سینٹینل", "sub": "تعمیراتی حفاظت اور پیشرفت کی AI نگرانی - تصاویر اور ویڈیوز",
    "photo": "تصویر جانچ", "video": "ویڈیو جانچ", "live": "لائیو", "zones": "علاقے",
    "progress": "پیشرفت", "report": "رپورٹ", "accuracy": "درستگی",
    "upload": "سائٹ کی تصویر یہاں ڈالیں", "sample": "یا نمونہ تصویر منتخب کریں",
    "run": "حفاظتی جانچ کریں", "blur": "چہرے دھندلے کریں (رازداری)",
    "model": "AI ماڈل", "accurate": "درست (yolo11m)", "fastm": "تیز (yolo11n)",
    "score": "حفاظتی اسکور", "workers": "مزدور", "helmets": "ہیلمٹ ٹھیک", "vests": "ویسٹ ٹھیک",
    "issues": "کن چیزوں پر توجہ درکار ہے", "none": "کوئی خلاف ورزی نہیں ملی۔ شاباش۔",
    "zone_hint": "جس شخص کے پاؤں ممنوعہ علاقے میں ہوں اسے نشان زد کیا جاتا ہے۔",
    "save_zone": "علاقہ شامل کریں", "clear_zone": "سب علاقے صاف کریں",
    "confirm": "مرحلہ کی تصدیق", "supervisor": "سپروائزر کا نام", "timeline": "جانچ کی تاریخ",
    "gen_report": "ہفتہ وار رپورٹ بنائیں", "estimate": "صرف تخمینہ - سائٹ سپروائزر سے تصدیق کروائیں۔",
    "site": "سائٹ کا نام", "orig": "اصل", "annot": "AI نتیجہ", "dets": "پہچان",
    "advice": "مشورہ", "vupload": "سائٹ کی ویڈیو یہاں ڈالیں (MP4, MOV, AVI, MKV, WEBM)",
    "vrun": "ویڈیو کا تجزیہ کریں", "rate": "فی سیکنڈ جانچے گئے فریم", "maxlen": "زیادہ سے زیادہ (منٹ)",
    "avg": "اوسط اسکور", "lowest": "کم ترین اسکور", "safe": "محفوظ وقت", "incidents": "واقعات",
    "maxw": "زیادہ سے زیادہ مزدور", "ptime": "پروسیسنگ وقت", "chart": "وقت کے ساتھ حفاظتی اسکور",
    "playfrom": "یہاں سے چلائیں", "dl_video": "ویڈیو ڈاؤن لوڈ کریں", "dl_csv": "ٹائم لائن (CSV)",
    "dl_json": "مکمل رپورٹ (JSON)", "noinc": "کوئی واقعہ نہیں - ہر جانچا گیا لمحہ محفوظ تھا۔",
    "vtip": "مشورہ: چھوٹی ویڈیوز (2 منٹ سے کم) لیپ ٹاپ پر جلدی مکمل ہوتی ہیں۔",
    "today": "اس سیشن کی جانچ", "avg_all": "اوسط اسکور", "inc_all": "ویڈیو واقعات",
}
for k, v in {"lang": "en", "timeline": [], "zones": [], "videos": [], "vstart": 0}.items():
    st.session_state.setdefault(k, v)
L = EN if st.session_state.lang == "en" else UR
RTL = st.session_state.lang == "ur"

# ---------------------------------------------------------------- theme ----
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
  #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden !important; }
  header[data-testid="stHeader"] { background: transparent !important; }
  html, body, .stApp { font-family: 'Inter', system-ui, sans-serif; }
  .stApp {
    background:
      radial-gradient(1200px 520px at 88% -12%, rgba(234,88,12,.16), transparent 60%),
      radial-gradient(900px 420px at -10% 18%, rgba(37,99,235,.10), transparent 60%),
      #070B14 !important; color:#E5E7EB; }
  .block-container { padding-top: 1rem; max-width: 1440px; }
  h1,h2,h3,h4,h5 { color:#F9FAFB !important; letter-spacing:-.02em; font-weight:800; }
  p, span, label, [data-testid="stCaptionContainer"] { color:#CBD5E1; }
  section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0B1220,#0A0F1C) !important; border-right:1px solid #1E293B; }
  section[data-testid="stSidebar"] * { color:#CBD5E1; }

  .stTextInput input, .stNumberInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
    background:#0F172A !important; color:#E5E7EB !important; border:1px solid #1F2937 !important; border-radius:12px !important; }
  ul[data-baseweb="menu"] { background:#0F172A !important; border:1px solid #1F2937; }
  ul[data-baseweb="menu"] li { color:#E5E7EB !important; }

  [data-testid="stFileUploaderDropzone"] { background: linear-gradient(160deg, rgba(234,88,12,.08), rgba(15,23,42,.9)) !important;
    border: 1.5px dashed rgba(251,146,60,.55) !important; border-radius: 18px !important; padding: 26px !important; }
  [data-testid="stFileUploaderDropzone"] button { background:#0F172A !important; color:#FDBA74 !important; border:1px solid rgba(251,146,60,.5) !important; border-radius:10px !important; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color:#FB923C !important; background: rgba(234,88,12,.12) !important; }

  .stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg,#EA580C,#F97316 55%,#FB923C) !important; color:#fff !important; border:none !important;
    border-radius:14px !important; font-weight:700 !important; box-shadow:0 6px 20px -6px rgba(234,88,12,.55) !important;
    transition: transform .12s ease, box-shadow .12s ease; }
  .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 10px 26px -6px rgba(234,88,12,.75) !important; }
  .stButton > button:disabled { background:#1F2937 !important; color:#64748B !important; box-shadow:none !important; }
  .ghost .stButton > button { background:#0F172A !important; border:1px solid #334155 !important; box-shadow:none !important; }

  .stTabs [role="tablist"] { background:rgba(15,23,42,.7); border-radius:16px; padding:6px; gap:6px; border:1px solid #1E293B;
    overflow-x:auto; margin-bottom:6px; }
  .stTabs [role="tab"] { color:#94A3B8 !important; border-radius:11px; padding:9px 16px !important; font-weight:600; transition: background .15s ease; }
  .stTabs [role="tab"]:hover { background:rgba(234,88,12,.10); color:#FDBA74 !important; }
  .stTabs [role="tab"][aria-selected="true"] { color:#fff !important; background:linear-gradient(135deg,#EA580C,#F97316);
    box-shadow:0 4px 14px -4px rgba(234,88,12,.6); }
  .stTabs [role="tab"] p { color:inherit !important; font-weight:600; }
  .stTabs [role="tab"][aria-selected="true"]::after, .stTabs [data-baseweb="tab-highlight-border"] { display:none !important; }

  .hero { position:relative; overflow:hidden; border-radius:26px; padding:26px 30px; margin-bottom:18px; border:1px solid #1F2937;
    background: linear-gradient(135deg, rgba(17,28,46,.95), rgba(15,23,42,.92) 55%, rgba(48,22,4,.9));
    box-shadow:0 24px 60px -30px rgba(0,0,0,.85), inset 0 1px 0 rgba(255,255,255,.05); }
  .hero::after { content:""; position:absolute; inset:0; pointer-events:none;
    background: radial-gradient(620px 220px at 92% 0%, rgba(234,88,12,.24), transparent 60%); }
  .hero::before { content:""; position:absolute; top:0; left:-60%; width:40%; height:100%; pointer-events:none;
    background:linear-gradient(100deg,transparent,rgba(255,255,255,.06),transparent); transform:skewX(-18deg); animation:shine 7s ease-in-out infinite; }
  @keyframes shine { 0%,60%{left:-60%;} 100%{left:130%;} }
  .hero-grid { display:flex; gap:24px; align-items:center; justify-content:space-between; flex-wrap:wrap; position:relative; z-index:1; }
  .hero h1 { margin:0; font-size:36px; }
  .hero p { margin:6px 0 0; color:#94A3B8; }
  .stat-row { display:flex; gap:12px; flex-wrap:wrap; }
  .stat { min-width:120px; padding:12px 16px; border-radius:16px; background:rgba(7,11,20,.55); border:1px solid #1F2937; backdrop-filter: blur(6px); }
  .stat b { display:block; font-size:24px; color:#fff; } .stat span { font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:#94A3B8; }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; background:rgba(234,88,12,.12); color:#FB923C;
    border:1px solid rgba(234,88,12,.35); margin:10px 6px 0 0; }
  .live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22C55E; margin-right:6px; box-shadow:0 0 0 0 rgba(34,197,94,.7); animation: pulse 1.8s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,.6);} 70%{box-shadow:0 0 0 10px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }

  .card { background:linear-gradient(160deg, rgba(17,28,46,.92), rgba(13,20,36,.92)); border:1px solid #1F2937; border-radius:20px;
    padding:18px 20px; height:100%; box-shadow:0 10px 30px -18px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.03);
    transition: transform .15s ease, border-color .15s ease; }
  .card:hover { transform:translateY(-2px); border-color:rgba(234,88,12,.45); }
  .card .lbl { color:#94A3B8; font-size:11px; text-transform:uppercase; letter-spacing:.12em; }
  .card .val { font-size:36px; font-weight:800; line-height:1.15; color:#F9FAFB; }
  .card .sub { font-size:12px; color:#64748B; margin-top:2px; }
  .ok{color:#22C55E !important;} .warn{color:#F59E0B !important;} .bad{color:#EF4444 !important;} .accent{color:#FB923C !important;}

  .ring-wrap { display:flex; align-items:center; gap:18px; }
  .ring { --p:0; --c:#22C55E; width:112px; height:112px; border-radius:50%; flex:none; display:grid; place-items:center;
    background: conic-gradient(var(--c) calc(var(--p)*1%), #1E293B 0); box-shadow:0 0 40px -12px var(--c); }
  .ring > div { width:86px; height:86px; border-radius:50%; background:#0B1220; display:grid; place-items:center; font-size:30px; font-weight:800; color:#fff; }

  .inc { border:1px solid #3B1F1F; background:linear-gradient(160deg, rgba(60,14,14,.45), rgba(13,20,36,.92)); border-radius:18px; padding:12px; margin-bottom:12px; }
  .inc .t { font-weight:800; color:#FCA5A5; font-size:15px; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px; margin:4px 4px 0 0; background:rgba(239,68,68,.14); color:#FCA5A5; border:1px solid rgba(239,68,68,.35); }
  .chip.okc { background:rgba(34,197,94,.12); color:#86EFAC; border-color:rgba(34,197,94,.35); }

  [data-testid="stImage"] img, video { border-radius:18px; border:1px solid #1F2937; box-shadow:0 16px 40px -20px rgba(0,0,0,.85); }
  [data-testid="stDataFrame"] { border-radius:14px; border:1px solid #1F2937; }
  .stAlert { border-radius:14px; }
  .rtl { direction: rtl; font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)


def score_cls(s) -> str:
    return "ok" if s >= 75 else ("warn" if s >= 55 else "bad")


def score_hex(s) -> str:
    return "#22C55E" if s >= 75 else ("#F59E0B" if s >= 55 else "#EF4444")


def card(lbl, val, cls="", sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div>{sub_html}</div>'


def ring_card(lbl, score, grade, sub=""):
    return (f'<div class="card"><div class="ring-wrap"><div class="ring" style="--p:{score};--c:{score_hex(score)}">'
            f'<div>{score:.0f}</div></div><div><div class="lbl">{lbl}</div>'
            f'<div class="val {score_cls(score)}" style="font-size:26px">{grade}</div>'
            f'<div class="sub">{sub}</div></div></div></div>')


def chips(counts: dict) -> str:
    if not counts:
        return '<span class="chip okc">no violations</span>'
    return "".join(f'<span class="chip">{k.replace("_", " ")} x{v}</span>' for k, v in counts.items())


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def T(text: str) -> str:
    """Wrap Urdu text right-to-left."""
    return f'<div class="rtl">{text}</div>' if RTL else text


# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown("### 🦺 Site Sentinel")
    st.caption("CPU-only · Apache-2.0 PPE model")
    lang = st.toggle("اردو / English", value=(st.session_state.lang == "ur"))
    if ("ur" if lang else "en") != st.session_state.lang:
        st.session_state.lang = "ur" if lang else "en"
        st.rerun()
    st.divider()
    site_name = st.text_input(L["site"], value="Demo Site")
    acc_ok, fast_ok = config.PPE_MODEL_ACCURATE.exists(), config.PPE_MODEL_FAST.exists()
    model_choice = st.radio(L["model"], [L["accurate"], L["fastm"]], index=0 if acc_ok else 1, horizontal=True)
    fast = model_choice == L["fastm"]
    if not acc_ok and not fast:
        st.caption("yolo11m not found - using yolo11n.")
    blur = st.checkbox(L["blur"], value=True)
    st.divider()
    st.caption("🎬 Video")
    sample_fps = st.slider(L["rate"], 0.5, 5.0, 2.0, 0.5)
    max_minutes = st.slider(L["maxlen"], 0.5, 10.0, 3.0, 0.5)
    st.divider()
    if config.METRICS_FILE.exists():
        m = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
        st.caption(f"📊 mAP@0.5 {m.get('map50')} · P {m.get('production', {}).get('precision')} "
                   f"· R {m.get('production', {}).get('recall')}")
    st.caption("⚠️ " + L["estimate"])

# ---------------------------------------------------------------- hero ----
tl = st.session_state.timeline
vids = st.session_state.videos
all_scores = [t["score"] for t in tl] + [v["report"].average_score for v in vids]
avg_all = f"{sum(all_scores) / len(all_scores):.0f}" if all_scores else "—"
inc_all = sum(len(v["report"].incidents) for v in vids)
active_model = model_path_for(fast).name
st.markdown(
    f'<div class="hero"><div class="hero-grid"><div>'
    f'<h1>🦺 {L["title"]}</h1><p>{L["sub"]}</p>'
    f'<span class="pill"><span class="live-dot"></span>AI ready · {active_model}</span>'
    f'<span class="pill">CPU</span><span class="pill">{site_name}</span></div>'
    f'<div class="stat-row">'
    f'<div class="stat"><b>{len(tl) + len(vids)}</b><span>{L["today"]}</span></div>'
    f'<div class="stat"><b>{avg_all}</b><span>{L["avg_all"]}</span></div>'
    f'<div class="stat"><b>{inc_all}</b><span>{L["inc_all"]}</span></div>'
    f'</div></div></div>', unsafe_allow_html=True)

tab_photo, tab_video, tab_live, tab_zones, tab_progress, tab_report, tab_bench = st.tabs([
    "🛡️ " + L["photo"], "🎬 " + L["video"], "🎥 " + L["live"], "🚧 " + L["zones"],
    "🏗️ " + L["progress"], "📄 " + L["report"], "📊 " + L["accuracy"]])

# ---------------------------------------------------------------- PHOTO ----
with tab_photo:
    c_up, c_opt = st.columns([2, 1])
    src = None
    with c_up:
        up = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png", "webp"])
    with c_opt:
        samples = sorted(glob.glob(str(config.SAMPLES_DIR / "*.jpg")) + glob.glob(str(config.SAMPLES_DIR / "*.png")))
        pick = st.selectbox(L["sample"], [Path(s).name for s in samples] or ["—"])
    if up is not None:
        src = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    elif samples and pick != "—":
        src = cv2.imread(str(config.SAMPLES_DIR / pick))

    if st.button("🛡️ " + L["run"], type="primary", disabled=src is None, width="stretch"):
        with st.spinner("Analysing photo…"):
            res = inspect_image(src, zones=st.session_state.zones, blur=blur, fast=fast)
        st.session_state.last = res
        st.session_state.last_orig = src
        st.session_state.timeline.append({"at": datetime.now().strftime("%H:%M:%S"), "score": res.score.score,
                                          "counts": dict(res.score.counts), "image": res.annotated})
        st.rerun()

    res = st.session_state.get("last")
    if res:
        s = res.score
        k1, k2, k3, k4 = st.columns([1.6, 1, 1, 1])
        k1.markdown(ring_card(L["score"], s.score, s.grade, f"{res.inference_ms:.0f} ms · {res.tiles_used} tile(s)"),
                    unsafe_allow_html=True)
        k2.markdown(card(L["workers"], s.workers, "accent"), unsafe_allow_html=True)
        k3.markdown(card(L["helmets"], s.compliant_helmets, "ok", f'{s.counts.get("missing_helmet", 0)} missing'),
                    unsafe_allow_html=True)
        k4.markdown(card(L["vests"], s.compliant_vests, "ok", f'{s.counts.get("missing_vest", 0)} missing'),
                    unsafe_allow_html=True)
        st.markdown(chips(s.counts), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.caption(L["orig"])
            if st.session_state.get("last_orig") is not None:
                st.image(rgb(st.session_state.last_orig), width="stretch")
        with right:
            st.caption(L["annot"])
            st.image(rgb(res.annotated), width="stretch")

        g1, g2, g3 = st.columns(3)
        if len(tl) >= 2:
            with g1:
                st.caption("Score trend")
                df = pd.DataFrame({"check": range(1, len(tl) + 1), "score": [t["score"] for t in tl]})
                st.altair_chart(alt.Chart(df).mark_area(line={"color": "#FB923C"}, color=alt.Gradient(
                    gradient="linear", stops=[alt.GradientStop(color="rgba(251,146,60,.45)", offset=0),
                                              alt.GradientStop(color="rgba(251,146,60,0)", offset=1)],
                    x1=1, x2=1, y1=0, y2=1)).encode(x="check:Q", y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]))),
                    height=180, width="stretch")
        if s.counts:
            with g2:
                st.caption("Violations")
                vc = pd.DataFrame([{"type": k.replace("_", " "), "count": v} for k, v in s.counts.items()])
                st.altair_chart(alt.Chart(vc).mark_bar(cornerRadius=6, color="#EF4444").encode(
                    x=alt.X("count:Q"), y=alt.Y("type:N", sort="-x"), tooltip=["type", "count"]), height=180, width="stretch")
        with g3:
            st.caption("Helmet compliance")
            donut = pd.DataFrame({"state": ["with helmet", "without"],
                                  "n": [s.compliant_helmets, s.counts.get("missing_helmet", 0)]})
            st.altair_chart(alt.Chart(donut).mark_arc(innerRadius=48, outerRadius=80).encode(
                theta="n", color=alt.Color("state", scale=alt.Scale(domain=["with helmet", "without"],
                                                                     range=["#22C55E", "#EF4444"])),
                tooltip=["state", "n"]), height=180, width="stretch")

        with st.expander("📋 " + L["dets"] + " & " + L["advice"]):
            for a in s.advice:
                st.markdown(T("• " + a), unsafe_allow_html=True)
            drows = [{"class": d.cls, "conf": round(d.conf, 2), "box": [int(v) for v in d.xyxy]}
                     for d in (res.detections or [])]
            if drows:
                st.dataframe(drows, width="stretch", hide_index=True, height=220)

        if len(tl) > 1:
            st.markdown("#### " + L["timeline"])
            idx = st.slider("check", 1, len(tl), len(tl)) - 1
            fr = tl[idx]
            st.caption(f"{fr['at']} · score {fr['score']}")
            st.image(rgb(fr["image"]), width="stretch")

# ---------------------------------------------------------------- VIDEO ----
with tab_video:
    st.caption("🎬 " + L["vtip"])
    vup = st.file_uploader(L["vupload"], type=[e.strip(".") for e in VIDEO_EXTENSIONS], key="vup")
    if vup is not None and st.button("🎬 " + L["vrun"], type="primary", width="stretch"):
        work = config.OUTPUT_DIR / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S")
        work.mkdir(parents=True, exist_ok=True)
        src_path = work / Path(vup.name).name
        src_path.write_bytes(vup.getvalue())
        bar = st.progress(0.0, text="Loading AI model…")
        try:
            rep = analyze_video(src_path, work, zones=st.session_state.zones, blur=blur, fast=fast,
                                sample_fps=sample_fps, max_seconds=max_minutes * 60,
                                progress=lambda f, msg: bar.progress(f, text=msg))
            st.session_state.videos.append({"name": vup.name, "report": rep,
                                            "at": datetime.now().strftime("%H:%M")})
            st.session_state.vsel = len(st.session_state.videos) - 1
            st.session_state.vstart = 0
            bar.progress(1.0, text=f"Done in {rep.processing_s:.0f} s")
            st.rerun()
        except Exception as e:  # unreadable file etc.
            bar.empty()
            st.error(f"Could not analyse this video: {e}")

    if st.session_state.videos:
        names = [f'{v["at"]} · {v["name"]}' for v in st.session_state.videos]
        sel = st.selectbox("Analysed videos", range(len(names)), format_func=lambda i: names[i],
                           index=st.session_state.get("vsel", len(names) - 1))
        st.session_state.vsel = sel
        rep = st.session_state.videos[sel]["report"]

        k = st.columns([1.6, 1, 1, 1, 1])
        k[0].markdown(ring_card(L["avg"], rep.average_score, rep.grade,
                                f"{rep.frames_analysed} frames · {mm_ss(rep.duration_s)}"), unsafe_allow_html=True)
        k[1].markdown(card(L["lowest"], rep.min_score, score_cls(rep.min_score), f"at {mm_ss(rep.worst_t)}"),
                      unsafe_allow_html=True)
        k[2].markdown(card(L["safe"], f"{rep.safe_time_pct:.0f}%", score_cls(rep.safe_time_pct)), unsafe_allow_html=True)
        k[3].markdown(card(L["incidents"], len(rep.incidents), "bad" if rep.incidents else "ok"), unsafe_allow_html=True)
        k[4].markdown(card(L["ptime"], f"{rep.processing_s:.0f}s", "accent", f"{L['maxw']}: {rep.max_workers}"),
                      unsafe_allow_html=True)

        vcol, icol = st.columns([1.7, 1])
        with vcol:
            st.video(rep.output_video, start_time=int(st.session_state.vstart))
            if not rep.browser_playable:
                st.caption("If the video does not play here, download it and open it in VLC.")
            df = pd.DataFrame([{"t": r.t, "score": r.score, "workers": r.workers,
                                "issues": ", ".join(f"{a.replace('_', ' ')} x{b}" for a, b in r.counts.items()) or "none"}
                               for r in rep.frames])
            st.markdown("#### 📈 " + L["chart"])
            base = alt.Chart(df).encode(x=alt.X("t:Q", title="seconds"))
            area = base.mark_area(line={"color": "#FB923C", "strokeWidth": 2}, color=alt.Gradient(
                gradient="linear", stops=[alt.GradientStop(color="rgba(251,146,60,.45)", offset=0),
                                          alt.GradientStop(color="rgba(251,146,60,0)", offset=1)],
                x1=1, x2=1, y1=0, y2=1)).encode(y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="score"),
                                                 tooltip=["t", "score", "workers", "issues"])
            layers = [area]
            if rep.incidents:
                inc_df = pd.DataFrame([{"start": i.start, "end": i.end, "types": ", ".join(i.types)} for i in rep.incidents])
                layers.insert(0, alt.Chart(inc_df).mark_rect(color="#EF4444", opacity=0.18).encode(
                    x="start:Q", x2="end:Q", tooltip=["start", "end", "types"]))
            layers.append(alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(strokeDash=[4, 4], color="#22C55E", opacity=.6)
                          .encode(y="y:Q"))
            st.altair_chart(alt.layer(*layers), height=220, width="stretch")

            if rep.violation_seconds:
                vs = pd.DataFrame([{"type": a.replace("_", " "), "seconds": b, "most at once": rep.most_at_once.get(a, 0)}
                                   for a, b in rep.violation_seconds.items()])
                st.altair_chart(alt.Chart(vs).mark_bar(cornerRadius=6, color="#EF4444").encode(
                    x=alt.X("seconds:Q", title="seconds with this issue"), y=alt.Y("type:N", sort="-x", title=None),
                    tooltip=["type", "seconds", "most at once"]), height=140, width="stretch")

        with icol:
            st.markdown(f"#### 🚨 {L['incidents']} ({len(rep.incidents)})")
            if not rep.incidents:
                st.success(L["noinc"])
            for n, inc in enumerate(rep.incidents, 1):
                st.markdown(f'<div class="inc"><div class="t">#{n} · {mm_ss(inc.start)} – {mm_ss(inc.end)} '
                            f'· score {inc.worst_score}</div>'
                            + "".join(f'<span class="chip">{t.replace("_", " ")}</span>' for t in inc.types)
                            + "</div>", unsafe_allow_html=True)
                if inc.snapshot and Path(inc.snapshot).exists():
                    st.image(inc.snapshot, width="stretch")
                if st.button(f"▶ {L['playfrom']} {mm_ss(inc.start)}", key=f"jump_{sel}_{n}"):
                    st.session_state.vstart = max(0, int(inc.start) - 1)
                    st.rerun()

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("⬇ " + L["dl_video"], Path(rep.output_video).read_bytes(),
                               file_name=Path(rep.output_video).name, mime="video/mp4", width="stretch")
        with d2:
            st.download_button("⬇ " + L["dl_csv"], Path(rep.timeline_csv).read_bytes(),
                               file_name=Path(rep.timeline_csv).name, mime="text/csv", width="stretch")
        with d3:
            st.download_button("⬇ " + L["dl_json"], Path(rep.report_json).read_bytes(),
                               file_name=Path(rep.report_json).name, mime="application/json", width="stretch")
        with d4:
            if rep.worst_frame and Path(rep.worst_frame).exists():
                st.download_button("⬇ Worst moment (JPG)", Path(rep.worst_frame).read_bytes(),
                                   file_name=Path(rep.worst_frame).name, mime="image/jpeg", width="stretch")

# ---------------------------------------------------------------- LIVE ----
with tab_live:
    st.caption("🎥 Real-time monitoring from your webcam or an RTSP camera. Uses the fast model for speed.")
    lc1, lc2 = st.columns([2, 1])
    with lc1:
        src_kind = st.selectbox("Source", ["Webcam 0", "RTSP URL", "Video file path"])
    with lc2:
        live_blur = st.checkbox("Blur faces", value=True, key="liveblur")
    if src_kind == "Video file path":
        target = st.text_input("Video path", "samples/site_demo.mp4")
    elif src_kind == "RTSP URL":
        target = st.text_input("RTSP URL", "rtsp://...")
    else:
        target = 0

    sc1, sc2, _ = st.columns([1, 1, 3])
    with sc1:
        started = st.button("▶ Start live", type="primary", disabled=st.session_state.get("live_running", False),
                            width="stretch")
    with sc2:
        stopped = st.button("⏹ Stop", disabled=not st.session_state.get("live_running", False), width="stretch")
    if started:
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            st.error("Could not open source. Check webcam / video path / RTSP URL.")
        else:
            st.session_state.cap = cap
            st.session_state.live_running = True
            st.session_state.live_scores = []
    if stopped and st.session_state.get("cap") is not None:
        st.session_state.cap.release()
        st.session_state.live_running = False

    lv1, lv2 = st.columns([2, 1])
    img_ph = lv1.empty()
    kpi_ph = lv2.empty()
    chart_ph = lv2.empty()

    @st.fragment(run_every=0.4)
    def _live_tick():
        if not st.session_state.get("live_running", False):
            return
        cap = st.session_state.get("cap")
        if cap is None or not cap.isOpened():
            st.session_state.live_running = False
            return
        ret, frame = cap.read()
        if not ret:
            st.session_state.live_running = False
            st.caption("Stream ended.")
            return
        res = inspect_image(frame, zones=st.session_state.zones, blur=live_blur, fast=True)
        st.session_state.live_scores.append(res.score.score)
        sc = res.score.score
        kpi_ph.markdown(ring_card("LIVE safety score", sc, res.score.grade, f"{res.score.workers} worker(s)")
                        + chips(res.score.counts), unsafe_allow_html=True)
        img_ph.image(rgb(res.annotated), width="stretch")
        if len(st.session_state.live_scores) >= 2:
            chart_ph.line_chart(pd.DataFrame({"score": st.session_state.live_scores[-120:]}), color="#FB923C", height=160)

    _live_tick()
    if st.session_state.get("live_running", False):
        st.markdown('<span class="pill"><span class="live-dot"></span>LIVE · ~2-3 fps on CPU</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------- ZONES ----
with tab_zones:
    st.markdown(T(L["zone_hint"]), unsafe_allow_html=True)
    zc1, zc2 = st.columns([1, 1.3])
    with zc1:
        zname = st.text_input("Zone name", "Excavation edge")
        st.caption("Corners as fractions of the image (0 = left/top, 1 = right/bottom).")
        default_pts = pd.DataFrame({"x": [0.05, 0.50, 0.50, 0.05], "y": [0.55, 0.55, 0.98, 0.98]})
        pts = st.data_editor(default_pts, num_rows="dynamic", width="stretch", key="zone_pts",
                             column_config={"x": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
                                            "y": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01)})
        poly = [(float(r.x), float(r.y)) for r in pts.dropna().itertuples()]
        b1, b2 = st.columns(2)
        if b1.button("➕ " + L["save_zone"], width="stretch", disabled=len(poly) < 3):
            st.session_state.zones.append(Zone(name=zname, polygon=poly))
            st.rerun()
        if b2.button("🗑 " + L["clear_zone"], width="stretch"):
            st.session_state.zones = []
            st.rerun()
        for z in st.session_state.zones:
            st.markdown(f"- 🚧 **{z.name}** · {len(z.polygon)} points")
        if not st.session_state.zones:
            st.info("No zones yet.")
    with zc2:
        base_img = st.session_state.get("last_orig")
        if base_img is None:
            base_img = np.full((540, 960, 3), (32, 23, 15), np.uint8)
        preview = base_img.copy()
        draw_zones(preview, st.session_state.zones + ([Zone(name=zname + " (preview)", polygon=poly)] if len(poly) >= 3 else []))
        st.image(rgb(preview), caption="Zone preview", width="stretch")

# ---------------------------------------------------------------- PROGRESS ----
with tab_progress:
    if "milestones" not in st.session_state:
        st.session_state.milestones = build_checklist()
    ms = st.session_state.milestones
    rep_p = compute_progress(ms)
    pc0, pc1, pc2 = st.columns([1.2, 2, 1.3])
    with pc0:
        st.markdown(ring_card(L["progress"], rep_p.percent, f"{rep_p.percent:.0f}%",
                              "confirmed" if rep_p.confirmed else "estimate"), unsafe_allow_html=True)
    with pc1:
        for m in ms:
            m.done = st.checkbox(f"{m.label}  ({int(m.weight * 100)}%)" + (f" ✅ {m.confirmed_by}" if m.confirmed_by else ""),
                                 value=m.done, key=m.stage)
    with pc2:
        sup = st.text_input(L["supervisor"], key="sup")
        stage = st.selectbox(L["confirm"], [m.stage for m in ms])
        if st.button("✅ " + ("Confirm" if not RTL else "تصدیق"), width="stretch"):
            if not sup:
                st.warning("Enter supervisor name.")
            else:
                st.session_state.milestones = confirm(ms, stage, sup, done=True)
                st.rerun()
    st.caption(rep_p.note)

# ---------------------------------------------------------------- REPORT ----
with tab_report:
    photo_scores = [t["score"] for t in tl]
    video_scores = [v["report"].average_score for v in vids]
    st.markdown(f"Photo checks: **{len(tl)}** · Videos: **{len(vids)}** · Video incidents: **{inc_all}**")
    if st.button("📄 " + L["gen_report"], type="primary"):
        if not tl and not vids:
            st.warning("Run at least one photo or video check first.")
        else:
            scores = photo_scores + video_scores
            mins = [t["score"] for t in tl] + [v["report"].min_score for v in vids]
            tally: dict = {}
            for t in tl:
                for a, b in t.get("counts", {}).items():
                    tally[a] = tally.get(a, 0) + b
            for v in vids:
                for a, secs in v["report"].violation_seconds.items():
                    tally[a] = tally.get(a, 0) + len([i for i in v["report"].incidents if a in i.types])
            top = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
            top_issues = [f"{a.replace('_', ' ')}: {b}" for a, b in top[:4]] or ["None recorded"]
            rp = compute_progress(st.session_state.get("milestones", build_checklist()))
            end = datetime.now()
            summary = WeekSummary(
                site_name=site_name, period_start=(end - timedelta(days=7)).strftime("%Y-%m-%d"),
                period_end=end.strftime("%Y-%m-%d"), inspections=len(tl) + len(vids),
                average_score=sum(scores) / len(scores), worst_score=min(mins),
                total_violations=sum(tally.values()), top_issues=top_issues,
                progress_percent=rp.percent, progress_confirmed=rp.confirmed)
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            thumbs = []
            for v in vids:
                thumbs += [Path(i.snapshot) for i in v["report"].incidents if i.snapshot][:2]
                if v["report"].worst_frame:
                    thumbs.append(Path(v["report"].worst_frame))
            pdf = config.OUTPUT_DIR / f"weekly_{end.strftime('%Y%m%d_%H%M%S')}.pdf"
            try:
                build_pdf(summary, pdf, thumbnail_paths=[p for p in thumbs if p.exists()][:6] or None)
            except TypeError:
                build_pdf(summary, pdf)
            texts = build_texts(summary)
            st.success(f"PDF: {pdf.name}")
            st.download_button("⬇ Download PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
            st.code(texts["whatsapp"], language=None)
            with st.expander("Email text"):
                st.code(texts["email"], language=None)

# ---------------------------------------------------------------- ACCURACY ----
with tab_bench:
    st.markdown("#### " + L["accuracy"])
    st.caption("Ground truth: your bundled PPE dataset (Roboflow 20250731-PPE2286, CC BY 4.0). "
               "Refresh with `python scripts/evaluate.py`.")
    if config.METRICS_FILE.exists():
        m = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
        a1, a2, a3, a4 = st.columns(4)
        p = m.get("production", {})
        a1.markdown(card("mAP@0.5", m.get("map50"), "accent"), unsafe_allow_html=True)
        a2.markdown(card("Precision", p.get("precision"), "warn"), unsafe_allow_html=True)
        a3.markdown(card("Recall", p.get("recall"), "ok"), unsafe_allow_html=True)
        a4.markdown(card("F1", p.get("f1"), ""), unsafe_allow_html=True)
        st.caption(f"model {m.get('model')} · {m.get('images')} images · {m.get('ms_per_image')} ms/image (CPU)")
        st.json(m.get("per_class_ap", {}))
    else:
        st.warning("No metrics yet. Run: python scripts/evaluate.py")

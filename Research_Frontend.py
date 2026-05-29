import streamlit as st
import os, re, json
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── FastAPI Backend URL ──────────────────────────────────────────────────────
BACKEND_URL   = os.getenv("BACKEND_URL",   "http://127.0.0.1:8000")
APP_PASSWORD  = os.getenv("APP_PASSWORD",  "")

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM LIGHT THEME CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #2D3748;
}

/* Background Linear Gradient */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 40%, #E2E8F0 100%);
    min-height: 100vh;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));
    color: #6366F1;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    margin-bottom: 1rem;
    border: 1px solid rgba(99, 102, 241, 0.15);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-title span {
    background: linear-gradient(135deg, #4F46E5, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #64748B;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 0;
}

/* Cards (Glassmorphic) */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 2.2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-bottom: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    box-shadow: 0 20px 40px -15px rgba(99, 102, 241, 0.08);
    transform: translateY(-2px);
}

/* Input Fields */
.stTextArea textarea {
    background: rgba(248, 250, 252, 0.8) !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 16px !important;
    color: #1E293B !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.98rem !important;
    padding: 1.2rem !important;
    transition: all 0.25s ease !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
}

/* Premium Gradient Button */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #EC4899 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.9rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.25) !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 50%, #DB2777 100%) !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
}
.stDownloadButton > button:hover {
    border-color: #6366F1 !important;
    color: #6366F1 !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* Progress Tracker Styling */
.step {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.8rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.step-idle {
    background: #F8FAFC;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
}
.step-active {
    background: linear-gradient(135deg, #EEF2FF, #F5F3FF);
    color: #4F46E5;
    border: 1px solid rgba(79, 70, 229, 0.2);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.05);
}
.step-done {
    background: #F0FDF4;
    color: #15803D;
    border: 1px solid rgba(21, 128, 61, 0.15);
}

/* Stat Chips */
.chip {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.95rem;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.6rem;
    margin-bottom: 0.6rem;
    border: 1px solid transparent;
}
.chip-purple { background: #EEF2FF; color: #4F46E5; border-color: rgba(79, 70, 229, 0.15); }
.chip-pink   { background: #FDF2F8; color: #DB2777; border-color: rgba(219, 39, 119, 0.15); }
.chip-green  { background: #F0FDF4; color: #16A34A; border-color: rgba(22, 163, 74, 0.15); }

/* Custom Headings & Typography inside Report Card */
.report-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: #0F172A;
    margin-top: 0;
    margin-bottom: 1.5rem;
    font-weight: 700;
}
.report-h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    color: #4F46E5;
    border-bottom: 2px solid #EEF2FF;
    padding-bottom: 0.4rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 700;
}
.report-h3 {
    font-size: 1.15rem;
    color: #0F172A;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
    font-weight: 700;
}

/* Slider Custom Styles */
.stSlider {
    padding: 0 0.25rem;
}

/* Section Label */
.label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 0.25rem;
}

/* Hide Streamlit Default Headers/Footers */
#MainMenu, footer, header {
    visibility: hidden;
    height: 0;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def strip_md(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Strip image tags
    text = re.sub(r'<!--.*?-->', '', text)       # Strip comment tags
    text = re.sub(r'#{1,6}\s?', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',   r'\1', text)
    text = re.sub(r'`(.*?)`',     r'\1', text)
    return text

def make_pdf(content, query):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Check if we can load standard Windows system TrueType fonts for native Unicode support (Hindi, Chinese, etc.)
    font_added = False
    for font_path in [
        "C:\\Windows\\Fonts\\SegoeUI.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
        "C:\\Windows\\Fonts\\Calibri.ttf"
    ]:
        if os.path.exists(font_path):
            try:
                pdf.add_font("UnicodeFont", "", font_path)
                pdf.set_font("UnicodeFont", size=11)
                font_added = True
                break
            except Exception:
                pass
                
    if not font_added:
        pdf.set_font("Helvetica", "", 11)
        
    pdf.set_text_color(15, 23, 42)
    
    # Try bold title if custom font loaded, else standard Helvetica Bold
    if font_added:
        pdf.set_font("UnicodeFont", size=18)
    else:
        pdf.set_font("Helvetica", "B", 18)
        
    pdf.cell(0, 12, "Research Report", ln=True, align="C")
    
    if font_added:
        pdf.set_font("UnicodeFont", size=10)
    else:
        pdf.set_font("Helvetica", "", 10)
        
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, f"Query: {query[:90]}", ln=True, align="C")
    pdf.ln(6)
    pdf.set_text_color(51, 65, 85)
    
    if font_added:
        pdf.set_font("UnicodeFont", size=11)
    else:
        pdf.set_font("Helvetica", "", 11)
    
    # Strip any comment tags and local image tags from PDF output
    clean_content = re.sub(r'<!--.*?-->', '', content)
    clean_content = re.sub(r'!\[.*?\]\(.*?\)', '', clean_content)
    plain = strip_md(clean_content)
    
    for line in plain.split("\n"):
        if font_added:
            try:
                pdf.multi_cell(0, 8, line)
            except Exception:
                pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
        else:
            pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
            
    return bytes(pdf.output())

def display_report_elements(report_text):
    text = report_text.strip()
    text = re.sub(r'^#\s+(.+)$',   r'<div class="report-title">\1</div>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$',  r'<div class="report-h2">\1</div>',    text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'<div class="report-h3">\1</div>',    text, flags=re.MULTILINE)
    st.markdown(text, unsafe_allow_html=True)

STEPS = [
    ("🧠", "Planning Research Strategy"),
    ("🌐", "Searching the Web"),
    ("🔍", "Extracting Key Facts"),
    ("🗜️", "Compressing Knowledge"),
    ("⚖️",  "Evaluating Research"),
    ("✍️", "Writing Final Report"),
    ("✅", "Finalizing"),
]
NODE_MAP = {
    "user_query": 0, "web_search": 1, "extract_info": 2,
    "summarize_notes": 3, "evaluate": 4, "optimize_research": 4,
    "refine_query": 4, "research_writer": 5, "save_notes": 6,
}

def render_steps(active, done):
    html = ""
    for i, (icon, label) in enumerate(STEPS):
        if i in done:
            cls, em = "step-done",   "✅"
        elif i == active:
            cls, em = "step-active", "⏳"
        else:
            cls, em = "step-idle",   icon
        html += f'<div class="step {cls}">{em} &nbsp;{label}</div>'
    return html



# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════
if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.markdown("<div style='max-width:400px;margin:8rem auto;'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Research Assistant")
    pwd = st.text_input("Password", type="password", placeholder="Enter access password")
    if st.button("Login", use_container_width=True):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔬 Advanced AI Research Platform</div>
  <div class="hero-title">Intelligent <span>Research Assistant</span></div>
  <p class="hero-sub">Autonomous multi-agent platform delivering cited, visual reports.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin-bottom: 2.5rem;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1.8], gap="large")

with left:
    # ── Input Box Card ──
    query = st.text_area(
        label="What would you like to research today?",
        placeholder="e.g., Breakthroughs in Fusion Energy, History of Deep Learning, Climate Change Solutions...",
        height=250,
        key="query_input",
    )

    with st.expander("⚙️  Advanced Parameters"):
        max_iter = st.slider("Research Loop Depth", 1, 5, 3)
        st.caption("Higher depth triggers deeper reflection & more iterative search loops.")
        
        target_lang = st.selectbox(
            "Target Language", 
            ["English", "Hindi", "Spanish", "French", "German", "Chinese"],
            index=0
        )
        
        writing_style = st.selectbox(
            "Writing Tone & Style",
            ["Book-Style Detailed", "Academic Thesis", "Executive Summary", "Casual & Engaging"],
            index=0
        )
        
        target_len = st.selectbox(
            "Target Length",
            ["Deep Book (6+ paragraphs per section)", "Medium-depth (3-4 paragraphs)", "Concise Summary (1-2 paragraphs)"],
            index=1
        )

    go = st.button("🚀  Launch Agent", use_container_width=True)
    st.caption("⏱ Typical run: 2–4 min at depth 3 · Faster at depth 1–2")

    if st.session_state.get("report"):
        if st.button("🔄  New Research", use_container_width=True):
            for k in ["report", "query", "iters_used", "max_iter"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown('<div class="label">Agent Pipeline Status</div>', unsafe_allow_html=True)
    steps_html = render_steps(-1, set())
    steps_placeholder = st.empty()
    steps_placeholder.markdown(steps_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with right:
    if go:
        if not query.strip():
            st.warning("⚠️ Please specify a valid research topic.")
            st.stop()

        payload = {
            "query":           query.strip(),
            "target_language": target_lang,
            "writing_style":   writing_style,
            "target_length":   target_len,
            "max_iteration":   max_iter,
        }

        status_text       = st.empty()
        plan_placeholder  = st.empty()
        token_placeholder = st.empty()
        done_steps        = set()
        active_step       = -1
        final_state       = {}
        current_iter      = 0
        token_buffer: list = []
        token_count        = 0

        with st.spinner("🔬 Research is in Progress..."):
            try:
                # Check backend health first
                _headers = {"X-API-Key": APP_PASSWORD} if APP_PASSWORD else {}
                try:
                    requests.get(f"{BACKEND_URL}/health", timeout=3)
                except Exception:
                    st.error(f"❌ Cannot connect to FastAPI backend at {BACKEND_URL}. Please make sure it is running (`python run.py`).")
                    st.stop()

                # Stream SSE events from FastAPI
                with requests.post(
                    f"{BACKEND_URL}/research/stream",
                    json=payload,
                    headers=_headers,
                    stream=True,
                    timeout=600
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data:"):
                            continue
                        try:
                            chunk = json.loads(line[5:].strip())
                        except Exception:
                            continue

                        # Token-level streaming from research_writer
                        if chunk.get("type") == "token":
                            token_buffer.append(chunk.get("text", ""))
                            token_count += 1
                            if token_count % 30 == 0:
                                visible = "".join(token_buffer)[-400:]
                                token_placeholder.markdown(
                                    f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;'
                                    f'padding:0.8rem 1rem;font-size:0.82rem;color:#475569;line-height:1.6;">'
                                    f'✍️ <strong>Writing report…</strong><br><span style="font-family:monospace;">'
                                    f'{visible}</span></div>',
                                    unsafe_allow_html=True,
                                )
                            continue

                        if "error" in chunk:
                            err_msg = chunk["error"]
                            if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                                st.error("❌ OpenAI quota exhausted. Add billing credits at platform.openai.com, then retry.")
                            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                                st.error("❌ OpenAI rate limit hit. Wait a few minutes, then retry. Reduce loop depth if this repeats.")
                            elif "connect" in err_msg.lower() or "timeout" in err_msg.lower():
                                st.error("❌ Network error — connection timed out or refused. Check your internet and retry.")
                            else:
                                st.error(f"❌ Backend error: {err_msg}")
                            st.stop()

                        node_name    = chunk.get("node", "")
                        state_update = chunk.get("state", {})

                        if isinstance(state_update, dict):
                            final_state.update(state_update)

                        # Clear token stream display once writing is done
                        if node_name == "research_writer":
                            token_placeholder.empty()
                            token_buffer.clear()

                        # Show research plan once user_query completes
                        if node_name == "user_query" and isinstance(state_update, dict):
                            plan_items = state_update.get("plan", [])
                            if plan_items:
                                # Strip ": {query}" suffix that fallback plan appends
                                def _clean(item):
                                    return re.sub(rf':\s*{re.escape(query.strip())}$', '', item, flags=re.IGNORECASE).strip()
                                plan_md = "**Research Plan — sections being investigated:**\n" + \
                                          "\n".join(f"- {_clean(p)}" for p in plan_items)
                                plan_placeholder.info(plan_md)

                        # Track iteration counter
                        if isinstance(state_update, dict) and "iteration" in state_update:
                            current_iter = state_update["iteration"]

                        step_index = NODE_MAP.get(node_name, -1)
                        if step_index >= 0:
                            if active_step >= 0:
                                done_steps.add(active_step)
                            done_steps.discard(step_index)  # allow revisit during loops
                            active_step = step_index
                        steps_placeholder.markdown(
                            render_steps(active_step, done_steps),
                            unsafe_allow_html=True
                        )
                        if step_index >= 0:
                            loop_num   = min(current_iter + 1, max_iter)
                            iter_label = f"  ·  Loop {loop_num}/{max_iter}" if current_iter > 0 or node_name not in ("user_query", "research_writer", "save_notes") else ""
                            status_text.caption(f"⏳ {STEPS[step_index][1]}{iter_label}")

                for i in range(len(STEPS)):
                    done_steps.add(i)
                steps_placeholder.markdown(render_steps(-1, done_steps), unsafe_allow_html=True)
                status_text.empty()
                plan_placeholder.empty()

            except Exception as e:
                err_str = str(e)
                if "quota" in err_str.lower() or "insufficient_quota" in err_str:
                    st.error("❌ OpenAI quota exhausted. Add billing credits at platform.openai.com, then retry.")
                elif "rate limit" in err_str.lower() or "429" in err_str:
                    st.error("❌ OpenAI rate limit hit. Wait a few minutes, then retry.")
                elif "timeout" in err_str.lower() or "connect" in err_str.lower():
                    st.error("❌ Request timed out. Ensure the backend is running and your internet is stable.")
                else:
                    st.error(f"❌ Execution terminated: {e}")
                st.stop()

        # Retrieve the compiled report from the accumulated state
        report = final_state.get("final_answer", "")

        if not report:
            st.warning("Agent execution succeeded, but no report content was compiled.")
            st.stop()

        st.session_state["report"]     = report
        st.session_state["query"]      = query.strip()
        st.session_state["iters_used"] = final_state.get("iteration", 0)
        st.session_state["max_iter"]   = max_iter
            
        st.success("🎉 Research successfully completed!")

    # ── Display Report ──
    if st.session_state.get("report"):
        report = st.session_state["report"]
        q      = st.session_state.get("query", "")

        _plain_text  = re.sub(r'!\[.*?\]\(.*?\)', '', report)           # remove image tags
        _plain_text  = re.sub(r'\[.*?\]\(.*?\)', '', _plain_text)       # remove links
        _plain_text  = re.sub(r'[#*`>_~|]|<!--.*?-->', '', _plain_text) # remove md symbols
        _plain_text  = re.sub(r'https?://\S+', '', _plain_text)         # remove bare URLs
        words        = len(_plain_text.split())
        reading_time = max(1, words // 200)
        ref_match    = re.search(r'##\s+References\s*\n(.*?)(?=\n##|\Z)', report, re.DOTALL | re.IGNORECASE)
        sources      = len(re.findall(r'^\[\d+\]', ref_match.group(1), re.MULTILINE)) if ref_match else 0
        iters_used   = st.session_state.get("iters_used", 0)
        max_iter_val = st.session_state.get("max_iter", 3)

        # Premium stat badges
        st.markdown(f"""
        <div style="margin-bottom: 1.2rem;">
          <span class="chip chip-purple">📊 {words:,} words</span>
          <span class="chip chip-pink">⏱ {reading_time} min read</span>
          <span class="chip chip-green">🔗 {sources} sources</span>
          <span class="chip chip-purple">🔄 {iters_used}/{max_iter_val} loops</span>
        </div>""", unsafe_allow_html=True)

        # Content Card with beautifully rendered text and downloaded local images
        st.markdown('<div class="card" style="background:#FFFFFF; padding: 3rem;">', unsafe_allow_html=True)
        display_report_elements(report)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Export & Downloads ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="label">💾 Export Final Assets</div>', unsafe_allow_html=True)

        slug = re.sub(r'[^\w\s-]', '', q)[:40].strip().replace(" ", "_")
        base = f"research_report_{slug}"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📝 Export Markdown", report.encode("utf-8-sig"), f"{base}.md",
                               "text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Export Plain Text", strip_md(report).encode("utf-8-sig"), f"{base}.txt",
                               "text/plain", use_container_width=True)
        with c3:
            if PDF_AVAILABLE:
                try:
                    pdf_b = make_pdf(report, q)
                    st.download_button("📕 Export PDF", pdf_b, f"{base}.pdf",
                                       "application/pdf", use_container_width=True)
                except Exception as pe:
                    st.caption(f"PDF compilation error: {pe}")
            else:
                st.caption("Install `fpdf2` for PDF compilation support.")

    else:
        # Pre-execution guidance panel
        st.markdown("""
        <div class="card" style="text-align:center; padding: 5rem 2rem; background: rgba(255, 255, 255, 0.65);">
          <div style="font-size:4rem; margin-bottom:1.2rem;">🔬</div>
          <h3 style="color:#0F172A; font-size:1.5rem; font-weight:700; margin-bottom: 0.6rem;">Ready to Research</h3>
          <div style="color:#64748B; font-size:0.95rem; line-height:1.9; max-width: 480px; margin: 0 auto 1.5rem;">
            Type any topic on the left — a medical condition, historical event, technology,
            travel destination, how-to guide, or anything you're curious about — and hit
            <strong>Launch Agent</strong>.
          </div>
          <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap; font-size:0.82rem; color:#94A3B8; font-weight:600;">
            <span>🌍 Multi-source web search</span>
            <span>🧠 AI fact extraction</span>
            <span>✍️ Structured long-form report</span>
            <span>📎 Cited sources</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

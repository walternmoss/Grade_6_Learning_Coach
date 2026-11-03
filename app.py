"""
Coach George: AI Learning Coach for 6th Grade Online Homeschool (Connections Academy)
=====================================================================================
A single-file Streamlit application utilizing the official `google-genai` SDK.

Pedagogical Goals:
1. Reduce Friction & Avoidance: Breaks complex lessons into single-question steps.
2. Buffer Parent-Child Tension: Objective "Good Enough" (85%+ benchmark) signals to stop over-editing.
3. Overcome the Typing Bottleneck: Native voice recording (`st.audio_input`) in chat and drafting.
4. Bypass Copy-Paste Restrictions: Multimodal ingestion of PDFs, Word docs (.docx), .txt, and images.
"""

import hashlib
import io
import os
import time
from typing import Dict, List, Optional

import docx
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field


# =====================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# =====================================================================
st.set_page_config(
    page_title="Coach George | 6th Grade Learning Coach",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean, low-distraction reading environment */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    /* High-visibility celebratory "Good Enough" banner */
    .good-enough-banner {
        background-color: #d1fae5;
        border: 3px solid #10b981;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #065f46;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.18);
    }
    /* Gentle revision banner when below 85% */
    .almost-there-banner {
        background-color: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #92400e;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    /* Subtle card styling for lesson materials and timer */
    .coach-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. STRUCTURED OUTPUT SCHEMAS (PYDANTIC)
# =====================================================================
class VoiceDraftResult(BaseModel):
    """Schema for Mode 2: Voice-to-Draft output."""

    what_i_heard: str = Field(
        description="A faithful transcription or brief summary of what the student said in the audio."
    )
    sixth_grade_draft: str = Field(
        description=(
            "2 to 3 clear, complete 6th-grade sentences that directly answer the question "
            "using the student's own ideas and natural vocabulary."
        )
    )
    coach_encouragement: str = Field(
        description="One short, upbeat sentence praising a specific thought from the recording."
    )


class CriterionStatus(BaseModel):
    """Individual rubric requirement status."""

    requirement: str = Field(description="Short description of the assignment requirement.")
    met: bool = Field(description="True if the student's draft adequately satisfies this point.")
    note: str = Field(description="Brief, friendly explanation of how it was met or what is missing.")


class RubricEvaluation(BaseModel):
    """Schema for Mode 3: 'Good Enough' Rubric Check."""

    score_percentage: int = Field(
        description="Estimated completion score from 0 to 100 based on 6th-grade expectations."
    )
    meets_good_enough_benchmark: bool = Field(
        description="True if score_percentage is 85 or higher."
    )
    what_went_well: List[str] = Field(
        description="2 to 3 specific strengths in the student's draft."
    )
    criteria_breakdown: List[CriterionStatus] = Field(
        description="Checklist evaluation of the core assignment requirements."
    )
    single_actionable_fix: Optional[str] = Field(
        default=None,
        description=(
            "ONLY populate if score_percentage < 85. Provide the ONE simplest tweak needed "
            "to cross the 85% finish line. Leave empty/null if score_percentage >= 85."
        ),
    )
    note_for_learning_coach: str = Field(
        description=(
            "A calm, reassuring note for the parent/Learning Coach explaining why this draft "
            "is developmentally appropriate for a 6th grader and why over-editing should be avoided."
        )
    )


# =====================================================================
# 3. SYSTEM PROMPTS & PEDAGOGICAL PERSONAS
# =====================================================================
SOCRATIC_SYSTEM_INSTRUCTION = """
You are "Coach George", a warm, patient, slightly witty AI Learning Coach helping an almost 12-year-old
6th-grade boy who attends Connections Academy online homeschool.

YOUR CORE RULES:
1. NEVER GIVE THE RAW ANSWER IMMEDIATELY: Your job is to help him discover the answer without feeling overwhelmed.
2. POINT HIS EYES TO THE CLUE: Whenever lesson materials (PDFs, Word worksheets, text, or screenshots) are loaded,
   direct his attention to the exact location of the clue (e.g., "Check out paragraph 2 under the 'Photosynthesis'
   heading..." or "Look at the arrows on the left side of the diagram...").
3. ONE QUESTION AT A TIME: Keep your responses short (under 90 words). End with ONLY ONE clear, low-stress question
   so his brain never feels crowded.
4. VALIDATE SPOKEN INPUT: If he sends an audio clip, briefly acknowledge what he said so he knows you heard him
   clearly, then guide him to the next tiny step.
5. COMBAT TASK AVOIDANCE: If he says "I don't know", "this is too long", or "I'm stuck", shrink the task into a
   multiple-choice or fill-in-the-blank micro-step.
"""

VOICE_TO_DRAFT_INSTRUCTION = """
You are "Coach George", assisting a 6th-grade student who thinks much faster than he types.
Listen carefully to his spoken recording. He may ramble, pause, repeat himself, or say "um" while thinking out loud.

YOUR TASK:
1. Capture what he said in `what_i_heard`.
2. Transform his raw thoughts into `sixth_grade_draft`: 2 to 3 clear, complete, grammatically sound sentences
   ready to paste into his Connections Academy worksheet or online quiz box.
3. CRITICAL VOICE RULE: Preserve his natural 11/12-year-old vocabulary and original ideas! Do NOT make him sound
   like a textbook, college student, or AI robot. Keep it authentic, clear, and 6th-grade appropriate.
4. If lesson materials or a target question are provided, ensure the 2-3 sentences accurately address the question
   using his spoken thoughts.
"""

RUBRIC_CHECK_INSTRUCTION = """
You are "Coach George", an objective, supportive 6th-grade evaluator acting as a healthy buffer between a
hard-working 6th grader and a well-meaning parent (Learning Coach) who sometimes pushes too hard for perfection.

YOUR GRADING PHILOSOPHY:
1. Evaluate against realistic 6th-grade standards—NOT adult or high-school honors perfection.
2. Minor spelling hiccups, slightly awkward phrasing, or simple sentence structures are 100% FINE as long as the
   core understanding and prompt requirements are met.
3. If the draft meets 85% or more of the core criteria (`score_percentage >= 85`), set `meets_good_enough_benchmark`
   to True and DO NOT suggest any rewrites (`single_actionable_fix` must be null/empty).
4. If the draft is below 85%, identify ONLY the single easiest addition or fix needed to cross the 85% threshold.
   Never pile on multiple criticisms.
"""


# =====================================================================
# 4. SESSION STATE INITIALIZATION
# =====================================================================
def init_session_state() -> None:
    """Initialize all Streamlit session state variables safely."""
    defaults = {
        "api_key": "",
        "selected_model": "gemini-2.5-flash",
        # Multimodal lesson cache: dict keyed by filename
        # Value: {"name": str, "type_label": str, "part": types.Part, "preview": str}
        "lesson_materials": {},
        # Mode 1 Chat history: list of {"role": "user"|"model", "display_text": str, "is_voice": bool}
        "chat_history": [
            {
                "role": "model",
                "display_text": (
                    "Hey there! I'm **Coach George** 🦉. Upload your Connections Academy lesson or "
                    "screenshot in the sidebar, and tell me (or record!) which question we're tackling first."
                ),
                "is_voice": False,
            }
        ],
        "last_chat_audio_hash": None,
        # Mode 2 Voice-to-Draft state
        "last_draft_audio_hash": None,
        "voice_transcript": "",
        "current_draft_text": "",
        "draft_coach_note": "",
        "draft_question_context": "",
        # Mode 3 Rubric Check state
        "rubric_prompt_text": "",
        "rubric_draft_text": "",
        "last_rubric_result": None,
        # Focus Sprint Timer (15 minutes = 900 seconds)
        "timer_running": False,
        "timer_end_ts": None,
        "timer_remaining_sec": 15 * 60,
        # Dopamine Progress Counter
        "questions_completed": 0,
        "daily_goal": 5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =====================================================================
# 5. API KEY & GENAI CLIENT HELPER
# =====================================================================
def resolve_api_key() -> str:
    """
    Retrieve the Gemini API key from environment variables, Streamlit secrets,
    or the user's sidebar session input.
    """
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if secret_key:
                return secret_key
    except Exception:
        # st.secrets raises an exception if no secrets.toml exists locally
        pass

    return st.session_state.get("api_key", "").strip()


def get_genai_client() -> Optional[genai.Client]:
    """Instantiate and return a configured `google.genai.Client`, or None if no key is available."""
    api_key = resolve_api_key()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        st.sidebar.error(f"Could not initialize Gemini client: {exc}")
        return None


# =====================================================================
# 6. MULTIMODAL LESSON INGESTION ENGINE
# =====================================================================
def extract_docx_text(file_bytes: bytes, filename: str) -> str:
    """
    Parse both paragraphs and table cells from a .docx worksheet into clean,
    structured text so Gemini can reference specific worksheet sections.
    """
    doc = docx.Document(io.BytesIO(file_bytes))
    output_lines: List[str] = [f"=== WORD WORKSHEET: {filename} ==="]

    # 1. Extract paragraphs
    para_count = 0
    for para in doc.paragraphs:
        cleaned = para.text.strip()
        if cleaned:
            para_count += 1
            style_name = para.style.name if para.style else "Normal"
            if "Heading" in style_name:
                output_lines.append(f"\n## {cleaned}")
            else:
                output_lines.append(f"[P{para_count}] {cleaned}")

    # 2. Extract tables (worksheets frequently use tables for graphic organizers)
    if doc.tables:
        output_lines.append("\n--- WORKSHEET TABLES / GRAPHIC ORGANIZERS ---")
        for t_idx, table in enumerate(doc.tables, start=1):
            output_lines.append(f"\n[Table {t_idx}]")
            for r_idx, row in enumerate(table.rows, start=1):
                # Deduplicate merged cells in the same row
                row_cells: List[str] = []
                for cell in row.cells:
                    cell_text = " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
                    if not row_cells or cell_text != row_cells[-1]:
                        row_cells.append(cell_text if cell_text else "(blank)")
                output_lines.append(f"  Row {r_idx}: " + " | ".join(row_cells))

    return "\n".join(output_lines)


def ingest_uploaded_file(uploaded_file) -> Dict:
    """
    Convert an uploaded file (.pdf, .docx, .txt, .png, .jpg, .jpeg) into a
    `google.genai.types.Part` object ready for multimodal inference.
    """
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    raw_bytes = uploaded_file.getvalue()

    if ext == ".docx":
        parsed_text = extract_docx_text(raw_bytes, filename)
        part = types.Part.from_text(text=parsed_text)
        return {
            "name": filename,
            "type_label": "Word Worksheet (.docx)",
            "part": part,
            "preview": parsed_text[:600] + ("..." if len(parsed_text) > 600 else ""),
        }

    if ext == ".txt":
        decoded_text = raw_bytes.decode("utf-8", errors="replace")
        labeled_text = f"=== TEXT LESSON: {filename} ===\n{decoded_text}"
        part = types.Part.from_text(text=labeled_text)
        return {
            "name": filename,
            "type_label": "Text Note (.txt)",
            "part": part,
            "preview": decoded_text[:600] + ("..." if len(decoded_text) > 600 else ""),
        }

    if ext == ".pdf":
        part = types.Part.from_bytes(data=raw_bytes, mime_type="application/pdf")
        return {
            "name": filename,
            "type_label": f"PDF Lesson ({len(raw_bytes) // 1024} KB)",
            "part": part,
            "preview": f"PDF binary loaded ({len(raw_bytes):,} bytes). Ready for visual & text reading.",
        }

    if ext in {".png", ".jpg", ".jpeg"}:
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        part = types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
        return {
            "name": filename,
            "type_label": f"Screenshot/Diagram ({ext})",
            "part": part,
            "preview": f"Image loaded ({mime_type}, {len(raw_bytes) // 1024} KB).",
            "raw_bytes": raw_bytes,
        }

    raise ValueError(f"Unsupported file extension: {ext}")


def get_active_lesson_parts() -> List[types.Part]:
    """Return a list of `types.Part` objects for all currently cached lesson files."""
    parts: List[types.Part] = []
    for item in st.session_state.lesson_materials.values():
        parts.append(
            types.Part.from_text(
                text=f"Attached Lesson Material: {item['name']} ({item['type_label']})"
            )
        )
        parts.append(item["part"])
    return parts


# =====================================================================
# 7. CLIPBOARD HELPER COMPONENT
# =====================================================================
def render_copy_button(text_to_copy: str, button_label: str = "📋 Copy Draft to Clipboard") -> None:
    """Render a clean one-click HTML/JS clipboard button for easy pasting into school portals."""
    safe_js_text = (
        text_to_copy.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", "\\n")
        .replace("\r", "")
    )
    html_code = f"""
    <div style="margin: 4px 0 12px 0;">
        <button id="copy-btn" onclick="copyText()" style="
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(37, 99, 235, 0.25);
            transition: background-color 0.2s;
        ">{button_label}</button>
        <span id="copy-status" style="margin-left: 10px; color: #059669; font-weight: 600; font-family: sans-serif;"></span>
    </div>
    <script>
    function copyText() {{
        const text = `{safe_js_text}`;
        navigator.clipboard.writeText(text).then(() => {{
            const status = document.getElementById("copy-status");
            const btn = document.getElementById("copy-btn");
            btn.style.backgroundColor = "#059669";
            btn.innerText = "✅ Copied! Ready to paste into Connections Academy";
            status.innerText = "";
            setTimeout(() => {{
                btn.style.backgroundColor = "#2563eb";
                btn.innerText = "{button_label}";
            }}, 3000);
        }});
    }}
    </script>
    """
    components.html(html_code, height=55)


# =====================================================================
# 8. SIDEBAR: API KEY, FOCUS SPRINT TIMER, & LESSON UPLOADER
# =====================================================================
def render_focus_timer() -> None:
    """
    Render a 15-minute Focus Sprint countdown timer in the sidebar.
    Uses `@st.fragment` when available so ticking every second does not interrupt
    audio recording or typing in the main workspace.
    """
    st.markdown("### ⏱️ 15-Min Focus Sprint")
    st.caption("Work hard for 15 minutes, then take a guilt-free break!")

    if st.session_state.timer_running and st.session_state.timer_end_ts:
        remaining = max(0, int(st.session_state.timer_end_ts - time.time()))
        st.session_state.timer_remaining_sec = remaining
        if remaining == 0:
            st.session_state.timer_running = False
            st.balloons()
            st.success("🎉 15-Minute Sprint Complete! Take a 5-minute stretch break!")

    mins, secs = divmod(st.session_state.timer_remaining_sec, 60)
    progress_ratio = 1.0 - (st.session_state.timer_remaining_sec / (15 * 60))
    st.progress(min(max(progress_ratio, 0.0), 1.0))
    st.markdown(
        f"<h2 style='text-align:center; margin: 0.2rem 0; font-family: monospace;'>{mins:02d}:{secs:02d}</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if not st.session_state.timer_running:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.timer_running = True
                st.session_state.timer_end_ts = (
                    time.time() + st.session_state.timer_remaining_sec
                )
                st.rerun()
        else:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.timer_running = False
                st.rerun()
    with col2:
        if st.button("🔄 15m", use_container_width=True, help="Reset to 15 minutes"):
            st.session_state.timer_running = False
            st.session_state.timer_remaining_sec = 15 * 60
            st.session_state.timer_end_ts = None
            st.rerun()
    with col3:
        if st.button("⚡ 5m", use_container_width=True, help="Quick 5-minute mini-sprint"):
            st.session_state.timer_running = False
            st.session_state.timer_remaining_sec = 5 * 60
            st.session_state.timer_end_ts = None
            st.rerun()


# Wrap the timer in a fragment if supported by the installed Streamlit version
if hasattr(st, "fragment"):
    @st.fragment(run_every=1.0)
    def sidebar_timer_fragment() -> None:
        render_focus_timer()
else:
    def sidebar_timer_fragment() -> None:
        render_focus_timer()


def render_sidebar() -> None:
    """Render sidebar controls: API key fallback, model selector, lesson uploader, and sprint tools."""
    with st.sidebar:
        st.title("🦉 Coach George")
        st.caption("Your 6th-Grade Connections Academy Sidekick")

        # --- API Key Configuration ---
        active_key = resolve_api_key()
        if not active_key:
            st.warning("🔑 Please enter your Gemini API Key to wake up Coach George.")
            entered_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Stored only in your browser session.",
            )
            if entered_key:
                st.session_state.api_key = entered_key.strip()
                st.rerun()
        else:
            with st.expander("⚙️ AI Brain Settings", expanded=False):
                st.success("API Key Connected ✅")
                st.session_state.selected_model = st.selectbox(
                    "Gemini Model",
                    options=["gemini-2.5-flash", "gemini-2.5-pro"],
                    index=0 if st.session_state.selected_model == "gemini-2.5-flash" else 1,
                    help=(
                        "gemini-2.5-flash is fast and great for everyday coaching. "
                        "gemini-2.5-pro is ideal for dense multi-page packets."
                    ),
                )
                if not os.environ.get("GEMINI_API_KEY"):
                    if st.button("Change API Key"):
                        st.session_state.api_key = ""
                        st.rerun()

        st.divider()

        # --- Multimodal Lesson Ingestion ---
        st.markdown("### 📚 Step 1: Load Your Lesson")
        st.caption(
            "Can't copy-paste from Connections Academy? Print the page to PDF, "
            "upload the `.docx` worksheet, or drop a screenshot here!"
        )

        uploaded_files = st.file_uploader(
            "Upload Lesson PDFs, Worksheets, or Screenshots",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for up_file in uploaded_files:
                if up_file.name not in st.session_state.lesson_materials:
                    try:
                        ingested = ingest_uploaded_file(up_file)
                        st.session_state.lesson_materials[up_file.name] = ingested
                    except Exception as exc:
                        st.error(f"Could not read `{up_file.name}`: {exc}")

        # Display currently cached materials in session_state
        if st.session_state.lesson_materials:
            st.success(
                f"📎 **{len(st.session_state.lesson_materials)} file(s) loaded in memory:**"
            )
            for fname, meta in list(st.session_state.lesson_materials.items()):
                with st.expander(f"📄 {fname}", expanded=False):
                    st.caption(meta["type_label"])
                    if "raw_bytes" in meta:
                        st.image(meta["raw_bytes"], caption=fname, use_container_width=True)
                    else:
                        st.text(meta["preview"])
            if st.button("🗑️ Clear Loaded Lesson Files", use_container_width=True):
                st.session_state.lesson_materials = {}
                st.rerun()
        else:
            st.info("No lesson files loaded yet. You can still chat or use Voice-to-Draft anytime!")

        st.divider()

        # --- 15-Minute Focus Sprint Timer ---
        sidebar_timer_fragment()

        st.divider()

        # --- Dopamine Question Counter ---
        st.markdown("### 🏆 Questions Crushed Today")
        done = st.session_state.questions_completed
        goal = st.session_state.daily_goal
        st.progress(min(done / max(goal, 1), 1.0))
        st.write(f"**{done} / {goal}** questions completed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Finished One!", use_container_width=True):
                st.session_state.questions_completed += 1
                if st.session_state.questions_completed == st.session_state.daily_goal:
                    st.balloons()
                st.rerun()
        with c2:
            if st.button("Reset Count", use_container_width=True):
                st.session_state.questions_completed = 0
                st.rerun()


# =====================================================================
# 9. MODE 1: SOCRATIC DETECTIVE (STEP-BY-STEP MULTIMODAL CHAT)
# =====================================================================
def generate_socratic_reply(
    client: genai.Client,
    user_text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
) -> str:
    """
    Send the active lesson materials, conversation history, and current user text/audio
    to Gemini using the Socratic Detective system prompt.
    """
    contents: List[types.Content] = []

    # 1. Attach active multimodal lesson materials first so the model can see them
    lesson_parts = get_active_lesson_parts()
    if lesson_parts:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Here are the student's current lesson materials and worksheets. "
                            "Refer to specific headings, paragraphs, tables, or diagram areas "
                            "when guiding him."
                        )
                    ),
                    *lesson_parts,
                ],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Got it! I have the lesson materials right in front of me and will guide him one step at a time."
                    )
                ],
            )
        )

    # 2. Replay prior chat history (excluding the brand-new message we are about to send)
    for turn in st.session_state.chat_history[:-1]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn["display_text"])],
            )
        )

    # 3. Build the current user turn (text, audio, or both)
    current_parts: List[types.Part] = []
    if audio_bytes:
        current_parts.append(
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        )
        current_parts.append(
            types.Part.from_text(
                text=(
                    "The student recorded this voice message. First, briefly confirm what he asked or "
                    "said in 1 short sentence, then guide him Socratically with ONE simple question."
                )
            )
        )
    if user_text:
        current_parts.append(types.Part.from_text(text=user_text))

    contents.append(types.Content(role="user", parts=current_parts))

    response = client.models.generate_content(
        model=st.session_state.selected_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SOCRATIC_SYSTEM_INSTRUCTION,
            temperature=0.5,
        ),
    )
    return response.text or "I'm right here! Which part of the lesson should we look at together?"


def render_socratic_mode(client: Optional[genai.Client]) -> None:
    """Render Mode 1: Socratic Detective Chat with quick-start buttons and native voice input."""
    st.subheader("🕵️‍♂️ Socratic Detective: Find the Clues Together")
    st.caption(
        "Coach George won't just hand you a wall of text—he'll point you to the exact spot in your lesson "
        "and ask **one short question at a time**."
    )

    # Quick-action buttons to overcome initial task avoidance / blank-page paralysis
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompt: Optional[str] = None
    with q_col1:
        if st.button("🗺️ Give me the 30-second tour", use_container_width=True):
            quick_prompt = (
                "Give me a super short, 3-bullet summary of what this lesson/worksheet is about, "
                "then ask me if I'm ready for Question 1."
            )
    with q_col2:
        if st.button("🎯 Help me start Question #1", use_container_width=True):
            quick_prompt = (
                "Help me start the very first question on my worksheet. Point me to where the clue "
                "lives in the lesson and ask me one easy warm-up question."
            )
    with q_col3:
        if st.button("🧩 Shrink this into a hint", use_container_width=True):
            quick_prompt = (
                "I'm feeling stuck. Can you turn this step into a super simple multiple-choice "
                "or fill-in-the-blank hint for me?"
            )
    with q_col4:
        if st.button("🧹 Reset Chat", use_container_width=True):
            st.session_state.chat_history = [
                {
                    "role": "model",
                    "display_text": (
                        "Fresh start! 🦉 Tell me or record which question we're looking at now."
                    ),
                    "is_voice": False,
                }
            ]
            st.session_state.last_chat_audio_hash = None
            st.rerun()

    # Render conversation history inside a scrollable container
    chat_container = st.container(height=420, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar = "🦉" if msg["role"] == "model" else "🧑‍💻"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg.get("is_voice"):
                    st.caption("🎙️ *Sent via voice recording*")
                st.markdown(msg["display_text"])

    # Bottom input row: Voice Recorder + Typed Chat Input
    voice_col, text_col = st.columns([1, 2])
    with voice_col:
        chat_audio = st.audio_input(
            "Record your thoughts (Chat)",
            key="socratic_audio_input",
            help="Talk to Coach George instead of typing!",
        )
    with text_col:
        st.write("")  # Vertical alignment spacer
        typed_message = st.chat_input(
            "Type a question number, paste a question, or say 'I'm stuck'..."
        )

    # Determine if new voice audio was recorded
    new_audio_bytes: Optional[bytes] = None
    if chat_audio is not None:
        raw_audio = chat_audio.getvalue()
        audio_hash = hashlib.md5(raw_audio).hexdigest()
        if audio_hash != st.session_state.last_chat_audio_hash:
            st.session_state.last_chat_audio_hash = audio_hash
            new_audio_bytes = raw_audio

    active_user_text = quick_prompt or typed_message

    if active_user_text or new_audio_bytes:
        if not client:
            st.error("Please add your Gemini API Key in the sidebar first!")
            return

        display_label = active_user_text if active_user_text else "🎙️ *[Spoken voice thought]*"
        st.session_state.chat_history.append(
            {
                "role": "user",
                "display_text": display_label,
                "is_voice": bool(new_audio_bytes),
            }
        )

        with st.spinner("Coach George is checking your lesson..."):
            try:
                reply = generate_socratic_reply(
                    client=client,
                    user_text=active_user_text,
                    audio_bytes=new_audio_bytes,
                )
                st.session_state.chat_history.append(
                    {
                        "role": "model",
                        "display_text": reply,
                        "is_voice": False,
                    }
                )
                st.rerun()
            except APIError as api_err:
                st.error(f"Gemini API Error: {api_err}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")


# =====================================================================
# 10. MODE 2: VOICE-TO-DRAFT (SPEAK IT -> 6TH-GRADE SENTENCES)
# =====================================================================
def transform_voice_to_draft(
    client: genai.Client,
    audio_bytes: bytes,
    question_context: str,
    include_lesson: bool,
) -> VoiceDraftResult:
    """
    Send the student's spoken audio (`audio/wav`) to Gemini and receive a structured
    `VoiceDraftResult` containing a faithful transcript and a 2-3 sentence 6th-grade draft.
    """
    parts: List[types.Part] = []

    if include_lesson and st.session_state.lesson_materials:
        parts.extend(get_active_lesson_parts())

    if question_context.strip():
        parts.append(
            types.Part.from_text(
                text=f"Worksheet Question / Prompt the student is answering:\n{question_context.strip()}"
            )
        )

    parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
    parts.append(
        types.Part.from_text(
            text=(
                "Listen to the student's spoken explanation. Turn his exact ideas into 2 to 3 clear, "
                "complete 6th-grade sentences while keeping his natural voice and vocabulary."
            )
        )
    )

    response = client.models.generate_content(
        model=st.session_state.selected_model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=VOICE_TO_DRAFT_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=VoiceDraftResult,
            temperature=0.3,
        ),
    )

    if response.parsed:
        return response.parsed
    return VoiceDraftResult.model_validate_json(response.text)


def render_voice_to_draft_mode(client: Optional[genai.Client]) -> None:
    """Render Mode 2: Voice-to-Draft converter with editable text area and 1-click clipboard copy."""
    st.subheader("🎙️ Voice-to-Draft: Talk Out Loud, Skip the Typing Freeze")
    st.caption(
        "Staring at a blank box? Just hit the microphone and explain your answer like you're talking "
        "to a friend. Coach George will tidy your thoughts into **2–3 complete 6th-grade sentences** "
        "using your own words!"
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 1️⃣ Tell Coach George Your Thoughts")
        question_ctx = st.text_input(
            "Which question are you answering? (Optional)",
            value=st.session_state.draft_question_context,
            placeholder="e.g., Question 4: Why do tectonic plates move?",
        )
        st.session_state.draft_question_context = question_ctx

        use_lesson_ctx = st.checkbox(
            "Include loaded sidebar lesson files for spelling/key terms",
            value=bool(st.session_state.lesson_materials),
            disabled=not bool(st.session_state.lesson_materials),
        )

        draft_audio = st.audio_input(
            "Record your thoughts",
            key="voice_to_draft_audio",
            help="Ramble, pause, or think out loud—Coach George will organize it!",
        )

        if draft_audio is not None:
            raw_audio = draft_audio.getvalue()
            audio_hash = hashlib.md5(raw_audio).hexdigest()

            # Automatically process a newly recorded clip, or allow manual re-generation
            trigger_transform = audio_hash != st.session_state.last_draft_audio_hash
            if st.button("✨ Turn My Voice into a 2–3 Sentence Draft", use_container_width=True):
                trigger_transform = True

            if trigger_transform:
                if not client:
                    st.error("Please enter your Gemini API Key in the sidebar first!")
                else:
                    st.session_state.last_draft_audio_hash = audio_hash
                    with st.spinner("Listening to your ideas and building your draft..."):
                        try:
                            result = transform_voice_to_draft(
                                client=client,
                                audio_bytes=raw_audio,
                                question_context=question_ctx,
                                include_lesson=use_lesson_ctx,
                            )
                            st.session_state.voice_transcript = result.what_i_heard
                            st.session_state.current_draft_text = result.sixth_grade_draft
                            st.session_state.draft_coach_note = result.coach_encouragement
                            st.rerun()
                        except APIError as api_err:
                            st.error(f"Gemini API Error: {api_err}")
                        except Exception as exc:
                            st.error(f"Could not process voice recording: {exc}")

        if st.session_state.voice_transcript:
            with st.expander("👂 What Coach George Heard You Say", expanded=True):
                st.write(st.session_state.voice_transcript)

    with col_right:
        st.markdown("#### 2️⃣ Your Polished 6th-Grade Draft")
        if st.session_state.draft_coach_note:
            st.success(f"🦉 **Coach George:** {st.session_state.draft_coach_note}")

        edited_draft = st.text_area(
            "Edit or tweak your sentences here before copying:",
            value=st.session_state.current_draft_text,
            height=160,
            placeholder="Your 2–3 sentence draft will appear here as soon as you record...",
        )
        st.session_state.current_draft_text = edited_draft

        if edited_draft.strip():
            # 1-Click Copy Button that copies the live edited text from st.text_area
            render_copy_button(edited_draft)

            st.caption("Or use the top-right copy icon in the box below:")
            st.code(edited_draft, language=None)

            if st.button(
                "➡️ Send This Draft to Mode 3 ('Good Enough' Check)",
                use_container_width=True,
            ):
                st.session_state.rubric_draft_text = edited_draft
                if st.session_state.draft_question_context and not st.session_state.rubric_prompt_text:
                    st.session_state.rubric_prompt_text = st.session_state.draft_question_context
                st.toast("Sent to 'Good Enough' Rubric Check! Click the Mode 3 tab above.", icon="✅")


# =====================================================================
# 11. MODE 3: "GOOD ENOUGH" RUBRIC CHECK (PARENT-CHILD BUFFER)
# =====================================================================
def evaluate_draft_against_rubric(
    client: genai.Client,
    assignment_prompt: str,
    student_draft: str,
    include_lesson: bool,
) -> RubricEvaluation:
    """
    Evaluate the student's draft against the assignment prompt/rubric using realistic
    6th-grade expectations (85%+ = Ready to Submit, no rewrites needed).
    """
    parts: List[types.Part] = []

    if include_lesson and st.session_state.lesson_materials:
        parts.extend(get_active_lesson_parts())

    parts.append(
        types.Part.from_text(
            text=(
                f"ASSIGNMENT PROMPT / RUBRIC:\n{assignment_prompt.strip() or 'Write a clear, accurate 6th-grade response based on the lesson.'}\n\n"
                f"STUDENT'S DRAFT:\n{student_draft.strip()}"
            )
        )
    )

    response = client.models.generate_content(
        model=st.session_state.selected_model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=RUBRIC_CHECK_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RubricEvaluation,
            temperature=0.2,
        ),
    )

    if response.parsed:
        return response.parsed
    return RubricEvaluation.model_validate_json(response.text)


def render_rubric_check_mode(client: Optional[genai.Client]) -> None:
    """Render Mode 3: Objective 'Good Enough' Rubric Check to prevent over-editing and burnout."""
    st.subheader("✅ 'Good Enough' Rubric Check (Objective Finish Line)")
    st.caption(
        "Perfectionism causes burnout! Paste the assignment instructions and your draft below. "
        "If it hits the **85% 6th-grade benchmark**, Coach George gives the official green light "
        "so everyone can close the laptop and move on."
    )

    col_in1, col_in2 = st.columns(2, gap="large")
    with col_in1:
        rubric_input = st.text_area(
            "📋 Assignment Prompt or Rubric Criteria",
            value=st.session_state.rubric_prompt_text,
            height=165,
            placeholder=(
                "Paste the question or rubric here (e.g., 'Explain 2 causes of the Dust Bowl "
                "in complete sentences using details from the lesson.')"
            ),
        )
        st.session_state.rubric_prompt_text = rubric_input

    with col_in2:
        draft_input = st.text_area(
            "✍️ Student's Draft",
            value=st.session_state.rubric_draft_text,
            height=165,
            placeholder="Paste the student's draft here (or send it from Mode 2)...",
        )
        st.session_state.rubric_draft_text = draft_input

    chk_col, btn_col = st.columns([2, 1])
    with chk_col:
        use_lesson_in_rubric = st.checkbox(
            "Also check against uploaded sidebar lesson/worksheet files",
            value=bool(st.session_state.lesson_materials),
            disabled=not bool(st.session_state.lesson_materials),
        )
    with btn_col:
        run_check = st.button(
            "🔍 Run 'Good Enough' Check",
            type="primary",
            use_container_width=True,
        )

    if run_check:
        if not client:
            st.error("Please enter your Gemini API Key in the sidebar first!")
            return
        if not draft_input.strip():
            st.warning("Please enter or paste the student's draft first!")
            return

        with st.spinner("Coach George is checking your draft against 6th-grade standards..."):
            try:
                evaluation = evaluate_draft_against_rubric(
                    client=client,
                    assignment_prompt=rubric_input,
                    student_draft=draft_input,
                    include_lesson=use_lesson_in_rubric,
                )
                st.session_state.last_rubric_result = evaluation
                if evaluation.score_percentage >= 85 or evaluation.meets_good_enough_benchmark:
                    st.balloons()
            except APIError as api_err:
                st.error(f"Gemini API Error: {api_err}")
            except Exception as exc:
                st.error(f"Could not evaluate draft: {exc}")

    # Display evaluation results if available
    result: Optional[RubricEvaluation] = st.session_state.last_rubric_result
    if result:
        st.divider()
        is_ready = result.score_percentage >= 85 or result.meets_good_enough_benchmark

        if is_ready:
            # Explicit green confirmation banner mandated by pedagogical spec
            st.success(
                "🎉 **Ready to submit! This meets all assignment criteria. Do not rewrite!**"
            )
            st.markdown(
                """
                <div class="good-enough-banner">
                    ✅ Ready to submit! This meets all assignment criteria. Do not rewrite!
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="almost-there-banner">
                    🛠️ Almost there! ({result.score_percentage}%) — Just ONE tiny tweak needed to cross the 85% finish line.
                </div>
                """,
                unsafe_allow_html=True,
            )

        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.metric(
                label="6th-Grade Readiness Score",
                value=f"{result.score_percentage}%",
                delta="Passes 85% Benchmark!" if is_ready else f"{85 - result.score_percentage}% to goal",
            )
            render_copy_button(
                st.session_state.rubric_draft_text,
                button_label="📋 Copy Final Draft",
            )

        with m_col2:
            st.markdown("#### 🌟 What You Nailed")
            for praise in result.what_went_well:
                st.markdown(f"- ✅ {praise}")

            if not is_ready and result.single_actionable_fix:
                st.info(f"🎯 **The ONLY Thing to Add/Fix:** {result.single_actionable_fix}")

        # Criteria Breakdown Table
        if result.criteria_breakdown:
            st.markdown("#### 📊 Rubric Checklist")
            for item in result.criteria_breakdown:
                icon = "🟢" if item.met else "🟡"
                st.markdown(f"{icon} **{item.requirement}** — {item.note}")

        # Dedicated Parent / Learning Coach Buffer Note
        st.markdown(
            f"""
            <div class="coach-card">
                <strong>☕ Note for the Learning Coach (Mom):</strong><br/>
                {result.note_for_learning_coach}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# 12. MAIN APPLICATION ENTRYPOINT
# =====================================================================
def main() -> None:
    """Main entrypoint for Coach George."""
    init_session_state()
    render_sidebar()

    client = get_genai_client()

    # Header bar
    st.markdown("## 🦉 Coach George — Your Calm 6th-Grade Study Partner")
    if st.session_state.lesson_materials:
        loaded_names = ", ".join(st.session_state.lesson_materials.keys())
        st.caption(f"🟢 **Active Lesson Material in Memory:** `{loaded_names}`")
    else:
        st.caption(
            "💡 *Tip: Upload your lesson PDF, Word worksheet (.docx), or screenshot in the left sidebar "
            "so Coach George can read along with you.*"
        )

    # Core Coaching Mode Tabs
    tab_socratic, tab_voice, tab_rubric = st.tabs(
        [
            "🕵️‍♂️ Mode 1: Socratic Detective (Chat)",
            "🎙️ Mode 2: Voice-to-Draft (Speak It)",
            "✅ Mode 3: 'Good Enough' Rubric Check",
        ]
    )

    with tab_socratic:
        render_socratic_mode(client)

    with tab_voice:
        render_voice_to_draft_mode(client)

    with tab_rubric:
        render_rubric_check_mode(client)


if __name__ == "__main__":
    main()
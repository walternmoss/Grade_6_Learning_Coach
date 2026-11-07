"""
Coach: Unified AI Learning Coach for 6th Grade Online Homeschool
=======================================================================
A single-screen, low-friction Streamlit app built with the `google-genai` SDK.

How It Works:
- One simple chat window supporting both Voice (`st.audio_input`) and Text (`st.chat_input`).
- Sidebar file drop for optional lesson materials (PDFs, Word worksheets, text, images).
- Coach automatically intuits what the student needs:
  1. Open-ended general knowledge answers (no document required!).
  2. Socratic guidance when working on specific school lessons.
  3. Automatic Voice/Thought-to-Draft (renders a copyable 2-3 sentence box in his own words).
  4. Automatic "Good Enough" (85%+) completion banner to prevent over-editing.
"""

import hashlib
import io
import os
from typing import Dict, List, Optional

import docx
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field


# =====================================================================
# 1. PAGE CONFIGURATION & MINIMALIST STYLING
# =====================================================================
st.set_page_config(
    page_title="Owl - Your AI Learning Coach",
    page_icon="🦉",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 860px;
    }
    /* Celebratory "Good Enough" banner rendered inside chat turns */
    .good-enough-banner {
        background-color: #d1fae5;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #065f46;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        margin: 0.75rem 0;
    }
    /* Reassuring note box for Mom / Learning Coach */
    .parent-note {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        color: #334155;
        font-size: 0.92rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. UNIFIED STRUCTURED OUTPUT SCHEMA
# =====================================================================
class CoachTurn(BaseModel):
    """
    Single structured response returned by Gemini on every turn.
    Allows Coach to seamlessly chat, answer general questions, draft sentences,
    or award the 'Good Enough' green light.
    """

    what_i_heard: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY if the student sent an audio recording. A brief, friendly 1-sentence "
            "summary or transcription of what he said so he knows you heard him."
        ),
    )
    coach_reply: str = Field(
        description=(
            "Your conversational reply to the student. Keep it warm, encouraging, and clear "
            "(under 120 words). IF HE ASKS A GENERAL KNOWLEDGE QUESTION (like history, science, "
            "the Talmud, religion, etc.), answer it directly and engagingly for an 11-year-old. "
            "NEVER ask for a document upload or school lesson when he asks a general knowledge question."
        )
    )
    ready_to_copy_draft: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY when the student explains an answer (by voice or rough text) or asks "
            "for help wording his thoughts for an assignment. Turn his exact ideas into 2 to 3 clear, complete "
            "6th-grade sentences preserving his natural 11/12-year-old vocabulary. "
            "Leave null/None if this is a general chat or question-answering turn."
        ),
    )
    is_good_enough_to_submit: bool = Field(
        default=False,
        description=(
            "Set to True ONLY when the student's draft meets 85% or more of an assignment's requirements."
        ),
    )
    note_for_learning_coach: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY when evaluating a draft or setting `is_good_enough_to_submit=True`. "
            "A brief, reassuring 1-2 sentence note for his Mom (Learning Coach)."
        ),
    )


# =====================================================================
# 3. UNIFIED SYSTEM INSTRUCTION (STRICTLY OPEN-ENDED FOR GENERAL Q&A)
# =====================================================================
UNIFIED_COACH_SYSTEM_PROMPT = """
You are "Coach", a warm, calm, slightly witty AI study partner for an almost 12-year-old
6th-grade boy attending Connections Academy online homeschool.

CORE RULES:
1. ABSOLUTELY OPEN-ENDED FOR GENERAL QUESTIONS:
   - If George asks about history, science, religion, the Talmud, ancient cultures, video games, 
     or anything else out of curiosity, ANSWER IT DIRECTLY AND AWESOMELY. 
   - DO NOT ask if he has his lesson open. DO NOT ask for a worksheet or document upload. 
     Just be an awesome, knowledgeable tutor explaining things clearly at an 11-year-old level.

2. IF HE IS WORKING ON A SCHOOL WORKSHEET / LESSON:
   - Only if files are uploaded or he specifically mentions homework/assignments, point his eyes 
     to the clue and keep guidance bite-sized and Socratic.

3. IF HE TALKS INTO THE MIC TO ANSWER A SCHOOL QUESTION:
   - Listen to his rambling thoughts, populate `what_i_heard`, and shape his ideas into `ready_to_copy_draft` 
     (2-3 clear 6th-grade sentences in his own words) with a one-click copy helper.

4. IF HE SHARES AN ASSIGNMENT DRAFT ("Is this good?"):
   - Grade against realistic 6th-grade expectations. If it hits ~85%+, set `is_good_enough_to_submit = True` 
     and give Mom the reassuring `note_for_learning_coach`.
"""


# =====================================================================
# 4. SESSION STATE INITIALIZATION
# =====================================================================
def init_session_state() -> None:
    """Initialize all required session state keys."""
    defaults = {
        "api_key": "",
        "selected_model": "gemini-2.5-flash",
        "lesson_materials": {},
        "last_audio_hash": None,
        "messages": [
            {
                "role": "model",
                "coach_reply": (
                    "Hey George! 🦉 Ask me anything—whether it's about history, science, the Talmud, "
                    "or a school worksheet. You can type or tap the microphone below!"
                ),
                "what_i_heard": None,
                "ready_to_copy_draft": None,
                "is_good_enough_to_submit": False,
                "note_for_learning_coach": None,
            }
        ],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =====================================================================
# 5. API KEY & GENAI CLIENT
# =====================================================================
def resolve_api_key() -> str:
    """Retrieve `GEMINI_API_KEY` from environment, `st.secrets`, or sidebar fallback."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if secret_key:
                return secret_key
    except Exception:
        pass

    return st.session_state.get("api_key", "").strip()


def get_genai_client() -> Optional[genai.Client]:
    """Return an initialized `google.genai.Client` or None if no key is set."""
    api_key = resolve_api_key()  # FIXED: Changed from get_api_key() to match function name
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    except Exception as exc:
        st.sidebar.error(f"Failed to initialize Gemini client: {exc}")
        return None


# =====================================================================
# 6. MULTIMODAL FILE INGESTION (.PDF, .DOCX, .TXT, .PNG, .JPG)
# =====================================================================
def extract_docx_text(file_bytes: bytes, filename: str) -> str:
    """Extract both paragraphs and table cells from a `.docx` worksheet."""
    doc = docx.Document(io.BytesIO(file_bytes))
    lines: List[str] = [f"=== WORKSHEET: {filename} ==="]

    p_idx = 0
    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            p_idx += 1
            lines.append(f"[P{p_idx}] {txt}")

    if doc.tables:
        lines.append("\n--- WORKSHEET TABLES ---")
        for t_idx, table in enumerate(doc.tables, start=1):
            lines.append(f"[Table {t_idx}]")
            for r_idx, row in enumerate(table.rows, start=1):
                cells: List[str] = []
                for cell in row.cells:
                    c_txt = " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
                    if not cells or c_txt != cells[-1]:
                        cells.append(c_txt if c_txt else "(blank)")
                lines.append(f"  Row {r_idx}: " + " | ".join(cells))

    return "\n".join(lines)


def ingest_uploaded_file(uploaded_file) -> Dict:
    """Convert an uploaded file into a `google.genai.types.Part`."""
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    raw_bytes = uploaded_file.getvalue()

    if ext == ".docx":
        text_content = extract_docx_text(raw_bytes, filename)
        return {
            "name": filename,
            "type_label": "Word Worksheet (.docx)",
            "part": types.Part.from_text(text=text_content),
        }

    if ext == ".txt":
        decoded = raw_bytes.decode("utf-8", errors="replace")
        return {
            "name": filename,
            "type_label": "Text File (.txt)",
            "part": types.Part.from_text(text=f"=== TEXT FILE: {filename} ===\n{decoded}"),
        }

    if ext == ".pdf":
        return {
            "name": filename,
            "type_label": "PDF Document",
            "part": types.Part.from_bytes(data=raw_bytes, mime_type="application/pdf"),
        }

    if ext in {".png", ".jpg", ".jpeg"}:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return {
            "name": filename,
            "type_label": "Screenshot / Image",
            "part": types.Part.from_bytes(data=raw_bytes, mime_type=mime),
        }

    raise ValueError(f"Unsupported file format: {ext}")


def build_lesson_parts() -> List[types.Part]:
    """Return a list of `types.Part` objects for all active uploaded files."""
    parts: List[types.Part] = []
    for meta in st.session_state.lesson_materials.values():
        parts.append(
            types.Part.from_text(text=f"Optional Uploaded Material: {meta['name']} ({meta['type_label']})")
        )
        parts.append(meta["part"])
    return parts


# =====================================================================
# 7. ONE-CLICK COPY BUTTON FOR DRAFTS
# =====================================================================
def render_copy_button(text_to_copy: str, unique_id: str) -> None:
    """Render a one-click clipboard copy button inside a chat bubble."""
    safe_text = (
        text_to_copy.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", "\\n")
        .replace("\r", "")
    )
    html_code = f"""
    <div style="margin: 4px 0;">
        <button id="btn-{unique_id}" onclick="copyDraft_{unique_id}()" style="
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        ">📋 Copy Sentences to Paste in School</button>
    </div>
    <script>
    function copyDraft_{unique_id}() {{
        navigator.clipboard.writeText(`{safe_text}`).then(() => {{
            const btn = document.getElementById("btn-{unique_id}");
            btn.style.backgroundColor = "#059669";
            btn.innerText = "✅ Copied! Now paste into your assignment";
            setTimeout(() => {{
                btn.style.backgroundColor = "#2563eb";
                btn.innerText = "📋 Copy Sentences to Paste in School";
            }}, 3000);
        }});
    }}
    </script>
    """
    components.html(html_code, height=45)


# =====================================================================
# 8. STREAMLINED SIDEBAR (OPTIONAL FILES ONLY)
# =====================================================================
def render_sidebar() -> None:
    """Render sidebar for optional file uploads and settings."""
    with st.sidebar:
        st.title("🦉 Owl - Your AI Learning Coach")
        st.caption("Chat freely, or drop a worksheet here if you want help with homework!")

        uploaded_files = st.file_uploader(
            "📎 (Optional) Add Lesson PDF or Worksheet",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for up_file in uploaded_files:
                if up_file.name not in st.session_state.lesson_materials:
                    try:
                        st.session_state.lesson_materials[up_file.name] = ingest_uploaded_file(
                            up_file
                        )
                    except Exception as exc:
                        st.error(f"Could not load {up_file.name}: {exc}")

        if st.session_state.lesson_materials:
            st.success(f"📚 {len(st.session_state.lesson_materials)} file(s) attached:")
            for fname, meta in st.session_state.lesson_materials.items():
                st.markdown(f"- **{fname}** *({meta['type_label']})*")

            if st.button("🗑️ Clear Attached Files", use_container_width=True):
                st.session_state.lesson_materials = {}
                st.rerun()

        st.divider()

        if st.button("🧹 Start a Brand New Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.last_audio_hash = None
            st.rerun()

        active_key = resolve_api_key()
        if not active_key:
            st.warning("🔑 Enter your Gemini API Key below to begin:")
            key_in = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
            if key_in:
                st.session_state.api_key = key_in.strip()
                st.rerun()
        else:
            with st.expander("⚙️ Settings", expanded=False):
                st.session_state.selected_model = st.selectbox(
                    "AI Model",
                    options=["gemini-2.5-flash", "gemini-2.5-pro"],
                    index=0 if st.session_state.selected_model == "gemini-2.5-flash" else 1,
                )


# =====================================================================
# 9. UNIFIED GEMINI CALL
# =====================================================================
def ask_coach(
    client: genai.Client,
    user_text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime_type: str = "audio/wav",
) -> CoachTurn:
    """Send history, optional files, and user input to Gemini, returning a `CoachTurn`."""
    contents: List[types.Content] = []

    # 1. Include optional uploaded lesson materials if present
    lesson_parts = build_lesson_parts()
    if lesson_parts:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Optional reference files attached by the student:"
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
                        text="Understood. I have these files available if needed for homework."
                    )
                ],
            )
        )

    # 2. Replay prior conversation history
    for msg in st.session_state.messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        summary_text = msg.get("coach_reply", "")
        if msg.get("ready_to_copy_draft"):
            summary_text += f"\n[Drafted Sentences: {msg['ready_to_copy_draft']}]"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=summary_text)],
            )
        )

    # 3. Append current turn
    current_parts: List[types.Part] = []
    if audio_bytes:
        current_parts.append(
            types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type)
        )
        current_parts.append(
            types.Part.from_text(
                text=(
                    "The student just recorded this voice message. Answer his question directly and warmly. "
                    "If he is talking out an assignment answer, format it into `ready_to_copy_draft`."
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
            system_instruction=UNIFIED_COACH_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=CoachTurn,
            temperature=0.4,
        ),
    )

    if response.parsed:
        return response.parsed
    return CoachTurn.model_validate_json(response.text)


# =====================================================================
# 10. MAIN SINGLE-SCREEN INTERFACE
# =====================================================================
def main() -> None:
    """Render the unified single-screen Coach interface."""
    init_session_state()
    render_sidebar()

    client = get_genai_client()

    # Header
    st.markdown("## 🦉 Owl- Your AI Learning Coach")
    if st.session_state.lesson_materials:
        names = ", ".join(st.session_state.lesson_materials.keys())
        st.caption(f"🟢 **Optional reference files attached:** `{names}`")
    else:
        st.caption("Ask any question or talk freely—no document upload required!")

    # Render Unified Chat Stream
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🦉" if msg["role"] == "model" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("what_i_heard"):
                st.caption(f"👂 *What I heard you say:* \"{msg['what_i_heard']}\"")

            st.markdown(msg["coach_reply"])

            if msg.get("ready_to_copy_draft"):
                st.markdown("##### ✍️ Your 6th-Grade Sentences (Ready to Copy):")
                updated_draft = st.text_area(
                    label="You can tweak any words here before copying:",
                    value=msg["ready_to_copy_draft"],
                    height=105,
                    key=f"draft_box_{idx}",
                )
                msg["ready_to_copy_draft"] = updated_draft
                render_copy_button(updated_draft, unique_id=f"turn_{idx}")

            if msg.get("is_good_enough_to_submit"):
                st.markdown(
                    """
                    <div class="good-enough-banner">
                        ✅ Ready to submit! This meets all assignment criteria. Do not rewrite!
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if msg.get("note_for_learning_coach"):
                st.markdown(
                    f"""
                    <div class="parent-note">
                        <strong>☕ Note for Learning Coach (Mom):</strong> {msg['note_for_learning_coach']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # --- Unified Bottom Input Bar (Voice + Text) ---
    voice_recording = st.audio_input(
        "🎙️ Tap the microphone to talk out loud (ask any question or speak your answer):",
        key="coach_voice_recorder",
    )

    typed_input = st.chat_input(
        "Or type/paste here (ask about the Talmud, history, science, or homework)..."
    )

    new_audio_bytes: Optional[bytes] = None
    audio_mime = "audio/wav"
    if voice_recording is not None:
        raw_audio = voice_recording.getvalue()
        current_hash = hashlib.md5(raw_audio).hexdigest()
        if current_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = current_hash
            new_audio_bytes = raw_audio
            audio_mime = getattr(voice_recording, "type", None) or "audio/wav"

    if new_audio_bytes is not None or typed_input:
        if not client:
            st.error("Please enter your Gemini API Key in the left sidebar first!")
            return

        user_display = typed_input if typed_input else "🎙️ *[Sent a voice recording]*"

        st.session_state.messages.append(
            {
                "role": "user",
                "coach_reply": user_display,
            }
        )

        with st.spinner("Coach is thinking..."):
            try:
                turn_result = ask_coach(
                    client=client,
                    user_text=typed_input,
                    audio_bytes=new_audio_bytes,
                    audio_mime_type=audio_mime,
                )

                st.session_state.messages.append(
                    {
                        "role": "model",
                        "coach_reply": turn_result.coach_reply,
                        "what_i_heard": turn_result.what_i_heard,
                        "ready_to_copy_draft": turn_result.ready_to_copy_draft,
                        "is_good_enough_to_submit": turn_result.is_good_enough_to_submit,
                        "note_for_learning_coach": turn_result.note_for_learning_coach,
                    }
                )

                if turn_result.is_good_enough_to_submit:
                    st.balloons()

                st.rerun()

            except APIError as api_err:
                st.error(f"Gemini API Error: {api_err}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()

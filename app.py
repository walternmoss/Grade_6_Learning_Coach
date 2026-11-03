"""
Coach: Unified AI Learning Coach for 6th Grade Online Homeschool
=======================================================================
A single-screen, low-friction Streamlit app built with the `google-genai` SDK.

How It Works:
- One simple chat window supporting both Voice (`st.audio_input`) and Text (`st.chat_input`).
- Sidebar file drop for Lesson PDFs, Word worksheets (.docx), text (.txt), and screenshots (.png/.jpg).
- Coach automatically intuits what the student needs on each turn:
  1. Socratic step-by-step guidance (pointing to where clues live in the uploaded files).
  2. Automatic Voice/Thought-to-Draft (renders a copyable 2-3 sentence box in his own words).
  3. Automatic "Good Enough" (85%+) completion banner to prevent over-editing and parent-child friction.
  4. Friendly clarification if a question prompt, worksheet, or rubric is missing.
"""

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
    page_title="Coach",
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
    Allows Coach to seamlessly chat, draft sentences, or award the
    'Good Enough' green light without making the student switch modes.
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
            "Your conversational reply to the student. Keep it warm, encouraging, and short "
            "(under 85 words). End with at most ONE clear question."
        )
    )
    ready_to_copy_draft: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY when the student explains an answer (by voice or rough text) or asks "
            "for help wording his thoughts. Turn his exact ideas into 2 to 3 clear, complete "
            "6th-grade sentences preserving his natural 11/12-year-old vocabulary. "
            "Leave null/None if you are still asking Socratic questions or clarifying the prompt."
        ),
    )
    is_good_enough_to_submit: bool = Field(
        default=False,
        description=(
            "Set to True ONLY when the student's draft (or the `ready_to_copy_draft` shaped from "
            "his spoken explanation) meets 85% or more of the assignment/question requirements "
            "at a realistic 6th-grade level."
        ),
    )
    note_for_learning_coach: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY when evaluating a draft or setting `is_good_enough_to_submit=True`. "
            "A brief, reassuring 1-2 sentence note for his Mom (Learning Coach) explaining why "
            "this meets 6th-grade standards and why no further rewrites are needed."
        ),
    )


# =====================================================================
# 3. UNIFIED SYSTEM INSTRUCTION
# =====================================================================
UNIFIED_COACH_SYSTEM_PROMPT = """
You are "Coach George", a warm, calm, slightly witty AI study partner for an almost 12-year-old
6th-grade boy attending Connections Academy online homeschool.

You operate in ONE unified conversation. Intuit what he needs based on what he sends you:

1. IF HE IS STUCK OR STARTING A LESSON / WORKSHEET:
   - Never dump a wall of text or hand him the raw answer unprompted.
   - Point his eyes to the exact spot in his uploaded lesson/screenshot (e.g., "Look at paragraph 2
     under the 'Water Cycle' heading..." or "Check out the middle row of the table...").
   - Ask ONLY ONE simple, low-stress question at a time.

2. IF HE TALKS INTO THE MIC (OR TYPES ROUGH IDEAS) TO ANSWER A QUESTION:
   - He thinks much faster than he types! Listen to his rambling spoken explanation or rough notes.
   - Populate `what_i_heard` so he sees you understood his voice clip.
   - Transform his thoughts into `ready_to_copy_draft`: 2 to 3 complete, grammatically sound 6th-grade
     sentences that he can copy-paste straight into Connections Academy.
   - CRITICAL: Keep his authentic 11/12-year-old voice and vocabulary. Never make him sound like a
     textbook or an adult.

3. IF HE SHARES A DRAFT OR ASKS "IS THIS GOOD ENOUGH?" (OR IF HIS SPOKEN ANSWER NAILS THE QUESTION):
   - Act as an objective buffer between him and a well-meaning parent (Learning Coach) who sometimes
     over-corrects or pushes for perfection.
   - Grade against realistic 6th-grade expectations. If his answer covers ~85%+ of what the question
     or rubric asks for, set `is_good_enough_to_submit = True` and include a reassuring
     `note_for_learning_coach` telling his Mom why it's ready to submit as-is.
   - If it's below 85%, DO NOT pile on critiques—point out the ONE simplest thing to add or tweak.

4. IF YOU ARE CONFUSED OR MISSING CONTEXT:
   - If he gives an answer or asks "Is this good?" but you don't have the worksheet question or rubric
     yet, warmly validate his effort and ask him to paste the question/rubric, upload the file in the
     sidebar, or read the question out loud into the mic.
"""


# =====================================================================
# 4. SESSION STATE INITIALIZATION
# =====================================================================
def init_session_state() -> None:
    """Initialize all required session state keys."""
    defaults = {
        "api_key": "",
        "selected_model": "gemini-2.5-flash",
        # Uploaded files stored once per session: {filename: {"type_label": str, "part": types.Part}}
        "lesson_materials": {},
        # Counter used to auto-reset the `st.audio_input` widget after each voice message
        "audio_widget_counter": 0,
        # Unified chat history
        "messages": [
            {
                "role": "model",
                "coach_reply": (
                    "Hey George! 🦉 Drop your lesson PDF, Word worksheet, or screenshot in the "
                    "sidebar—or just tap the microphone below and tell me what we're working on!"
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
    api_key = resolve_api_key()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
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
            types.Part.from_text(text=f"Uploaded Material: {meta['name']} ({meta['type_label']})")
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
# 8. STREAMLINED SIDEBAR (FILES + RESET ONLY)
# =====================================================================
def render_sidebar() -> None:
    """Render a distraction-free sidebar focused solely on uploading lesson files."""
    with st.sidebar:
        st.title("🦉 Coach George")
        st.caption("Upload your lesson or worksheet here, then chat or talk on the right!")

        uploaded_files = st.file_uploader(
            "📎 Add Lesson PDF, Worksheet (.docx), or Screenshot",
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
            st.success(f"📚 {len(st.session_state.lesson_materials)} file(s) loaded:")
            for fname, meta in st.session_state.lesson_materials.items():
                st.markdown(f"- **{fname}** *({meta['type_label']})*")

            if st.button("🗑️ Clear Uploaded Files", use_container_width=True):
                st.session_state.lesson_materials = {}
                st.rerun()

        st.divider()

        if st.button("🧹 Start a Brand New Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.audio_widget_counter += 1
            st.rerun()

        # API Key fallback & Model selector tucked neatly at the bottom
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
def ask_coach_george(
    client: genai.Client,
    user_text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
) -> CoachTurn:
    """
    Send active lesson files, conversation history, and the current voice/text input
    to Gemini, returning a structured `CoachTurn`.
    """
    contents: List[types.Content] = []

    # 1. Include uploaded lesson materials at the start of context
    lesson_parts = build_lesson_parts()
    if lesson_parts:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Here are the lesson files, worksheets, or screenshots currently loaded:"
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
                        text="I have read the uploaded lesson materials and am ready to help."
                    )
                ],
            )
        )

    # 2. Replay prior conversation history (excluding the newly appended user message)
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

    # 3. Append the current turn (Voice audio, typed text, or both)
    current_parts: List[types.Part] = []
    if audio_bytes:
        current_parts.append(
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        )
        current_parts.append(
            types.Part.from_text(
                text=(
                    "The student just recorded this voice message. Listen to what he is asking or "
                    "explaining. If he is answering a question out loud, shape his thoughts into "
                    "`ready_to_copy_draft` (2-3 6th-grade sentences in his own words) and evaluate "
                    "whether it meets the 85% `is_good_enough_to_submit` benchmark."
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
    st.markdown("## 🦉 Coach George")
    if st.session_state.lesson_materials:
        names = ", ".join(st.session_state.lesson_materials.keys())
        st.caption(f"🟢 **Reading along with:** `{names}`")
    else:
        st.caption(
            "Talk into the mic, type a question, or upload your worksheet in the left sidebar!"
        )

    # Render Unified Chat Stream
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🦉" if msg["role"] == "model" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            # If this is a model reply to a voice recording, show what Coach heard
            if msg.get("what_i_heard"):
                st.caption(f"👂 *What I heard you say:* \"{msg['what_i_heard']}\"")

            # Main conversational text
            st.markdown(msg["coach_reply"])

            # If Coach shaped his thoughts into a 2-3 sentence draft, show an editable + copyable card
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

            # If the answer hits the 85%+ benchmark, show the celebratory "Good Enough" banner
            if msg.get("is_good_enough_to_submit"):
                st.markdown(
                    """
                    <div class="good-enough-banner">
                        ✅ Ready to submit! This meets all assignment criteria. Do not rewrite!
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Optional reassuring note for Mom / Learning Coach
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
    # Dynamic key automatically clears the audio widget after each voice submission
    audio_key = f"voice_recorder_{st.session_state.audio_widget_counter}"
    voice_recording = st.audio_input(
        "🎙️ Tap the microphone to talk out loud (ask a question or speak your answer):",
        key=audio_key,
    )

    typed_input = st.chat_input(
        "Or type/paste here (a question number, your rough draft, or a rubric)..."
    )

    # Trigger a turn if the student recorded voice OR submitted text
    if voice_recording is not None or typed_input:
        if not client:
            st.error("Please enter your Gemini API Key in the left sidebar first!")
            return

        audio_bytes = voice_recording.getvalue() if voice_recording is not None else None
        user_display = (
            typed_input
            if typed_input
            else "🎙️ *[Sent a voice recording]*"
        )

        # 1. Append student's message to chat
        st.session_state.messages.append(
            {
                "role": "user",
                "coach_reply": user_display,
            }
        )

        # 2. Get structured reply from Coach George
        with st.spinner("Coach is thinking..."):
            try:
                turn_result = ask_coach_george(
                    client=client,
                    user_text=typed_input,
                    audio_bytes=audio_bytes,
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

                # Increment the audio widget key so the mic resets cleanly for the next turn
                if audio_bytes is not None:
                    st.session_state.audio_widget_counter += 1

                st.rerun()

            except APIError as api_err:
                st.error(f"Gemini API Error: {api_err}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()

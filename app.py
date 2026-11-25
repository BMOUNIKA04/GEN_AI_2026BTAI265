import os
import re
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# Page configuration
st.set_page_config(
    page_title="Chat with Research Paper",
    page_icon="📄",
    layout="centered",
)

MODEL_NAME = "gemini-2.5-flash"
MAX_PAPER_CHARS = 200_000
NOT_MENTIONED_REPLY = "This is not mentioned in the paper."

SYSTEM_INSTRUCTION = f"""You are a helpful academic research assistant.
Your job is to answer the user's questions strictly using ONLY the provided research paper text.

Rules:
1. Base every part of your answer ONLY on the text inside <paper>...</paper>.
2. Do NOT use outside knowledge, assumptions, or information not explicitly written in the paper.
3. If the paper does not contain the answer to the user's question, respond with EXACTLY:
{NOT_MENTIONED_REPLY}
4. When the answer is in the paper, keep it clear and concise, and cite the page number(s) like (Page 3) based on the [Page N] markers in the text.
"""


def get_default_api_key() -> str:
    """Return GEMINI_API_KEY from Streamlit secrets or environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def extract_pdf_pages(uploaded_file) -> tuple[int, list[tuple[int, str]]]:
    """Return total page count and extracted non-empty (page_num, text) pairs."""
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = re.sub(r"\s+\n", "\n", raw_text).strip()
        if cleaned_text:
            pages.append((page_num, cleaned_text))
    return len(reader.pages), pages


def select_paper_context(pages: list[tuple[int, str]], question: str) -> str:
    """Format paper pages with [Page N] markers, trimming if the PDF is unusually large."""
    formatted_pages = [f"[Page {num}]\n{text}" for num, text in pages]
    full_text = "\n\n".join(formatted_pages)
    if len(full_text) <= MAX_PAPER_CHARS:
        return full_text

    # For very large PDFs, keep the opening pages plus the highest keyword-matching pages.
    keywords = {w for w in re.findall(r"[a-zA-Z0-9]{3,}", question.lower())}
    scored_indices: list[tuple[int, int]] = []
    for idx, (_, text) in enumerate(pages):
        lowered = text.lower()
        score = sum(lowered.count(kw) for kw in keywords)
        if idx < 2:
            score += 1000  # Always prioritize title/abstract/intro pages
        scored_indices.append((score, idx))

    selected = set()
    char_budget = 0
    for _, idx in sorted(scored_indices, reverse=True):
        block = formatted_pages[idx]
        if char_budget + len(block) > MAX_PAPER_CHARS:
            continue
        selected.add(idx)
        char_budget += len(block)

    return "\n\n".join(formatted_pages[i] for i in sorted(selected))


def ask_paper_question(
    api_key: str,
    pages: list[tuple[int, str]],
    chat_history: list[dict[str, str]],
    question: str,
) -> str:
    """Send the paper text, recent chat context, and question to Gemini."""
    paper_context = select_paper_context(pages, question)

    # Include recent turns so follow-up questions work naturally.
    recent_history = chat_history[-6:]
    history_block = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
    )

    prompt_parts = [f"<paper>\n{paper_context}\n</paper>"]
    if history_block:
        prompt_parts.append(f"<conversation_history>\n{history_block}\n</conversation_history>")
    prompt_parts.append(f"Question: {question}")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="\n\n".join(prompt_parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        ),
    )
    return (response.text or "").strip() or NOT_MENTIONED_REPLY


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "paper_id" not in st.session_state:
        st.session_state.paper_id = None
    if "paper_pages" not in st.session_state:
        st.session_state.paper_pages = []
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


init_session_state()

st.title("📄 Chat with Research Paper")
st.caption(
    "Upload a PDF research paper and ask questions. Answers are grounded strictly in the paper."
)

# Sidebar: API key, PDF upload, paper info, and quick prompts
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=get_default_api_key(),
        type="password",
        help="Get a free API key from aistudio.google.com",
    )

    uploaded_file = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
    )

    if uploaded_file is not None:
        current_paper_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.paper_id != current_paper_id:
            with st.spinner("Reading PDF..."):
                try:
                    total_pages, pages = extract_pdf_pages(uploaded_file)
                    st.session_state.paper_id = current_paper_id
                    st.session_state.paper_pages = pages
                    st.session_state.total_pages = total_pages
                    st.session_state.messages = []
                except Exception as exc:
                    st.error(f"Could not read PDF: {exc}")
                    st.session_state.paper_id = None
                    st.session_state.paper_pages = []
                    st.session_state.total_pages = 0

        if st.session_state.paper_pages:
            word_count = sum(len(text.split()) for _, text in st.session_state.paper_pages)
            st.success(
                f"Loaded **{uploaded_file.name}**\n\n"
                f"- Pages with text: {len(st.session_state.paper_pages)} / {st.session_state.total_pages}\n"
                f"- Approx. words: {word_count:,}"
            )
        elif st.session_state.paper_id == current_paper_id:
            st.warning(
                "No selectable text was found in this PDF. It may be a scanned image PDF."
            )

    st.divider()
    st.subheader("Sample Questions")
    sample_questions = [
        "What dataset did they use?",
        "What is the main contribution of this paper?",
        "What evaluation metrics or results are reported?",
        "What limitations do the authors mention?",
    ]
    for sample in sample_questions:
        if st.button(sample, use_container_width=True, disabled=not st.session_state.paper_pages):
            st.session_state.pending_question = sample

    if st.button("Clear Chat History", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

# Render chat history on screen
if not st.session_state.paper_pages:
    st.info("👈 Upload a research paper (PDF) and enter your Gemini API key in the sidebar to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept input from either the chat box or a clicked sample question button
typed_question = st.chat_input("Ask a question about the paper (e.g., What dataset did they use?)")
user_question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if user_question:
    if not st.session_state.paper_pages:
        st.warning("Please upload a readable PDF research paper first.")
    elif not api_key.strip():
        st.warning("Please enter your Google Gemini API key in the sidebar.")
    else:
        prior_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the paper..."):
                try:
                    answer = ask_paper_question(
                        api_key=api_key.strip(),
                        pages=st.session_state.paper_pages,
                        chat_history=prior_history,
                        question=user_question,
                    )
                except Exception as exc:
                    answer = f"Error calling Gemini API: {exc}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

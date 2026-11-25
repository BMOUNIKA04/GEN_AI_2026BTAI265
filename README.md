# Chat with Research Paper (PDF Q&A Bot)

A simple Streamlit web app where students upload a research paper (PDF) and ask questions about it. The bot answers **strictly based on the uploaded paper** using Google's Gemini API (`gemini-2.5-flash`). If the answer is not in the paper, it replies: `"This is not mentioned in the paper."`

## Features

- **PDF Upload & Text Extraction**: Reads multi-page research papers page-by-page using `pypdf` and tags each page (`[Page 1]`, `[Page 2]`, etc.).
- **Paper-Only Grounded Answers**: Uses a strict system instruction and low temperature (`0.1`) so the model cites page numbers and avoids guessing outside the paper.
- **Persistent Chat History**: Displays the conversation on screen using `st.chat_message` and `st.session_state`.
- **Sample Questions & Large PDF Support**: One-click sample prompts in the sidebar and automatic keyword-based page selection if a PDF exceeds 200,000 characters.

## Setup & Run Locally

1. **Get a free Gemini API key** from [Google AI Studio](https://aistudio.google.com/).
2. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL shown in your terminal, paste your Gemini API key in the sidebar (or set the `GEMINI_API_KEY` environment variable), upload a PDF, and start asking questions!

## Free Hosting (Streamlit Community Cloud)

1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and deploy `app.py`.
3. Optionally add `GEMINI_API_KEY = "your_key"` in **App Settings → Secrets**, or enter your key in the sidebar when using the app.

## How the Code Works (Quick Walkthrough)

1. **`extract_pdf_pages()`**: Opens the uploaded PDF with `pypdf.PdfReader` and extracts text from every page, keeping track of page numbers.
2. **`select_paper_context()`**: Formats pages with `[Page N]` labels. For normal ~15-page papers, the entire paper is sent to Gemini; for very long PDFs, it keeps the opening pages plus the pages matching keywords from your question.
3. **`ask_paper_question()`**: Sends the paper context, recent chat messages, and your question to `gemini-2.5-flash` via the `google-genai` SDK with a strict `SYSTEM_INSTRUCTION` requiring paper-only answers.
4. **`st.session_state.messages`**: Stores the user and assistant messages so the chat history remains visible on screen across interactions.

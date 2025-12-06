"""
Streamlit LLaMA / ChatGPT-like Chat App
File: streamlit_llama_chat_app.py

Features:
- ChatGPT-like chat UI (message bubbles) using Streamlit
- Supports three model backends: OpenAI Chat API, llama-cpp-python (local LLaMA), or HuggingFace Transformers
- Upload documents (PDF, DOCX, TXT) and images (PNG, JPG). Extracts text and attaches it to prompt context.
- OCR for images using pytesseract (optional)
- Simple conversation memory in session_state
- Tooling functions to parse PDFs and DOCX

How to run:
1) Create virtual environment and install dependencies:
   pip install -r requirements.txt

Example requirements.txt contents:
streamlit
openai
llama-cpp-python  # optional, if you want local LLaMA via llama.cpp
transformers
torch
pdfplumber
python-docx
pytesseract
Pillow
tiktoken

2) Environment variables (if using OpenAI):
   export OPENAI_API_KEY="sk-..."

3) If using llama-cpp, have a local .bin/.ggml model and point to it in the UI when selecting backend.

Run:
   streamlit run streamlit_llama_chat_app.py

Notes:
- This example focuses on structure and integration points. Running heavy local models requires GPU/CPU resources and proper model files.
- The code tries to gracefully fall back when optional libraries are not present.

"""

import os
import streamlit as st
from typing import List, Tuple
import tempfile

# Optional imports are wrapped to avoid hard failures
try:
    import openai
except Exception:
    openai = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception:
    transformers = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------------------- Utility functions ----------------------

def init_session():
    if 'history' not in st.session_state:
        st.session_state.history = []  # list of (role, text)
    if 'backend' not in st.session_state:
        st.session_state.backend = 'openai'
    if 'model' not in st.session_state:
        st.session_state.model = ''


def append_history(role: str, text: str):
    st.session_state.history.append((role, text))


def clear_history():
    st.session_state.history = []


# ---------------------- File parsing helpers ----------------------

def extract_text_from_pdf(file_path: str) -> str:
    if pdfplumber is None:
        return ""  # library not installed
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_path: str) -> str:
    if docx is None:
        return ""
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def ocr_image(file_path: str) -> str:
    if pytesseract is None or Image is None:
        return ""  # OCR not available
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


# ---------------------- Model wrappers ----------------------

def openai_chat_completion(messages: List[dict], model: str = 'gpt-3.5-turbo') -> str:
    if openai is None:
        return "OpenAI library not installed or not available."
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        return "OPENAI_API_KEY not set."
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2, max_tokens=800)
        return resp['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenAI error: {e}"


def llama_cpp_completion(prompt: str, model_path: str, max_tokens: int = 256) -> str:
    if Llama is None:
        return "llama-cpp-python not installed."
    if not os.path.exists(model_path):
        return "Model file not found for llama-cpp."
    try:
        llm = Llama(model_path=model_path)
        out = llm(prompt=prompt, max_tokens=max_tokens)
        return out.get('choices', [{}])[0].get('text', '').strip()
    except Exception as e:
        return f"Llama-cpp error: {e}"


def hf_text_generation(prompt: str, model_name: str = 'gpt2', max_length: int = 256) -> str:
    if transformers is None:
        return "transformers is not installed."
    try:
        gen = pipeline('text-generation', model=model_name)
        out = gen(prompt, max_length=max_length, do_sample=False)
        return out[0]['generated_text']
    except Exception as e:
        return f"Transformers error: {e}"


# Build the messages payload for OpenAI style models
def build_messages_from_history(history: List[Tuple[str, str]], user_new_text: str, file_context: str = None) -> List[dict]:
    messages = []
    for role, text in history:
        messages.append({'role': role, 'content': text})
    user_content = user_new_text
    if file_context:
        user_content = f"{user_new_text}\n\n[Attached file content begins]\n{file_context}\n[Attached file content ends]"
    messages.append({'role': 'user', 'content': user_content})
    return messages


# ---------------------- Streamlit UI ----------------------

def sidebar_controls():
    st.sidebar.title('Settings & Backends')
    backend = st.sidebar.selectbox('Choose backend', options=['openai', 'llama_cpp', 'huggingface'], index=0)
    st.session_state.backend = backend

    if backend == 'openai':
        st.sidebar.write('OpenAI settings')
        model = st.sidebar.text_input('OpenAI model', value='gpt-3.5-turbo')
        st.session_state.model = model
        st.sidebar.info('Make sure OPENAI_API_KEY environment variable is set.')
    elif backend == 'llama_cpp':
        st.sidebar.write('Local LLaMA (llama-cpp) settings')
        model_path = st.sidebar.text_input('Path to .bin/.ggml model file (local)', value='')
        st.session_state.model = model_path
        st.sidebar.info('Install llama-cpp-python and have a local model file. e.g. ggml-model-q4_0.bin')
    else:
        st.sidebar.write('HuggingFace Transformers settings')
        hf_model = st.sidebar.text_input('HF model name', value='gpt2')
        st.session_state.model = hf_model
        st.sidebar.info('Large HF models require disk and memory. Consider using smaller models for testing.')

    if st.sidebar.button('Clear chat'):
        clear_history()


def render_chat():
    st.title('🔮 Chat — Streamlit LLaMA / ChatGPT-like')
    st.write('Upload documents or images and ask questions. The app will attach extracted text to your prompt.')

    # File uploaders and context extraction
    with st.expander('Upload files (PDF, DOCX, TXT)'):
        uploaded_files = st.file_uploader('Choose files', accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        file_context = []
        if uploaded_files:
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.' + f.name.split('.')[-1]) as tmp:
                    tmp.write(f.read())
                    tmp.flush()
                    suffix = f.name.split('.')[-1].lower()
                    if suffix == 'pdf':
                        txt = extract_text_from_pdf(tmp.name)
                    elif suffix == 'docx':
                        txt = extract_text_from_docx(tmp.name)
                    else:
                        txt = extract_text_from_txt(tmp.name)
                    file_context.append(f"----- FILE: {f.name} -----\n{txt}\n")
        file_context = '\n'.join(file_context) if file_context else None

    with st.expander('Upload images (optional OCR)'):
        uploaded_images = st.file_uploader('Images', accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        img_texts = []
        if uploaded_images:
            for im in uploaded_images:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.' + im.name.split('.')[-1]) as tmp:
                    tmp.write(im.read())
                    tmp.flush()
                    ocr_text = ocr_image(tmp.name)
                    img_texts.append(f"----- IMAGE: {im.name} (OCR) -----\n{ocr_text}\n")
        if img_texts:
            img_context = '\n'.join(img_texts)
            if file_context:
                file_context = file_context + '\n' + img_context
            else:
                file_context = img_context

    # Chat input
    user_input = st.text_area('Your message', height=120)

    col1, col2 = st.columns([1, 6])
    if col1.button('Send') or st.session_state.get('auto_send'):
        if user_input.strip() == '':
            st.warning('Please enter a message.')
        else:
            append_history('user', user_input)
            # Build prompt/messages
            backend = st.session_state.backend
            model = st.session_state.model
            # Display a temporary reply placeholder
            append_history('assistant', '...thinking...')

            # Prepare the prompt or messages depending on backend
            if backend == 'openai':
                messages = build_messages_from_history(st.session_state.history[:-1], user_input, file_context)
                response = openai_chat_completion(messages, model=model or 'gpt-3.5-turbo')
            elif backend == 'llama_cpp':
                # for llama-cpp we concatenate conversation into a single prompt
                convo = '\n'.join([f"{r.upper()}: {t}" for r, t in st.session_state.history[:-1]])
                full_prompt = convo + '\nASSISTANT:'
                if file_context:
                    full_prompt = full_prompt + '\n\nAttached File Context:\n' + file_context + '\n\n'
                full_prompt = full_prompt + '\nUSER: ' + user_input + '\nASSISTANT:'
                response = llama_cpp_completion(full_prompt, model_path=model)
            else:
                # HuggingFace text generation
                convo = '\n'.join([f"{r.upper()}: {t}" for r, t in st.session_state.history[:-1]])
                full_prompt = convo + '\nUSER: ' + user_input + '\nASSISTANT:'
                if file_context:
                    full_prompt = full_prompt + '\nAttached File Context:\n' + file_context
                response = hf_text_generation(full_prompt, model_name=model or 'gpt2')

            # Replace the '...thinking...' with actual response
            # pop last assistant placeholder then append real assistant
            if st.session_state.history and st.session_state.history[-1][1] == '...thinking...':
                st.session_state.history.pop()
            append_history('assistant', response)

    # Messages display
    st.markdown('---')
    for i, (role, text) in enumerate(st.session_state.history[-20:]):
        if role == 'user':
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    # Quick export
    if st.button('Export chat as TXT'):
        txt = '\n'.join([f"{r.upper()}: {t}" for r, t in st.session_state.history])
        st.download_button('Download', txt, file_name='chat_history.txt')


# ---------------------- Main ----------------------

def main():
    init_session()
    sidebar_controls()
    render_chat()


if __name__ == '__main__':
    main()

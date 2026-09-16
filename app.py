import os
import tempfile

import faiss
import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f4f8;
            margin-bottom: 20px;
        }

        .answer-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f8f9fa;
            border-left: 5px solid #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">👩‍💼 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your company HR policy using Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key."
    )

    st.markdown("---")

    st.markdown("### 📚 How it works")

    st.write(
        """
        1. Upload an HR Policy PDF
        2. Extract text from the PDF
        3. Split text into chunks
        4. Generate embeddings
        5. Store embeddings in FAISS
        6. Retrieve relevant policy sections
        7. Generate an answer using Groq
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.code("openai/gpt-oss-20b")


# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in document:
        page_text = page.get_text()

        if page_text:
            text += page_text + "\n"

    document.close()

    return text


# ---------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------

def create_chunks(
    text,
    chunk_size=800,
    overlap=150
):

    text = text.replace("\r", "\n")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# Create FAISS Index
# ---------------------------------------------------------

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # Normalize embeddings so inner product behaves
    # like cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Retrieve Relevant Chunks
# ---------------------------------------------------------

def retrieve_chunks(
    query,
    index,
    chunks,
    top_k=4
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx >= 0:

            results.append(
                {
                    "text": chunks[idx],
                    "score": float(score)
                }
            )

    return results


# ---------------------------------------------------------
# Generate Answer with Groq
# ---------------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    api_key
):

    client = Groq(
        api_key=api_key
    )

    context = "\n\n".join(
        [
            f"[Policy Section {i + 1}]\n{item['text']}"
            for i, item in enumerate(
                retrieved_chunks
            )
        ]
    )

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY using
the information provided in the HR policy context.

Rules:

1. Do not invent HR policies.
2. Do not make assumptions.
3. If the answer is not available in the provided
   policy context, clearly say that the information
   is not available in the uploaded HR policy.
4. Give concise and professional answers.
5. When useful, explain the relevant policy details.
6. Do not reveal or discuss the internal RAG process.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Answer the employee's question using only
the HR policy context above.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

st.subheader("📄 Upload HR Policy")

uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"]
)


# ---------------------------------------------------------
# Process PDF
# ---------------------------------------------------------

if uploaded_file:

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing HR policy..."
        ):

            extracted_text = (
                extract_text_from_pdf(
                    uploaded_file
                )
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in the PDF."
                )

                st.stop()

            chunks = create_chunks(
                extracted_text
            )

            if not chunks:

                st.error(
                    "Could not create text chunks from the PDF."
                )

                st.stop()

            index = create_faiss_index(
                chunks
            )

            st.session_state.index = index

            st.session_state.chunks = chunks

            st.session_state.processed_file = (
                uploaded_file.name
            )

            st.session_state.extracted_text = (
                extracted_text
            )

        st.success(
            f"HR Policy processed successfully! "
            f"Created {len(chunks)} searchable chunks."
        )

    else:

        st.success(
            f"Using processed policy: "
            f"{uploaded_file.name}"
        )


# ---------------------------------------------------------
# Example Questions
# ---------------------------------------------------------

st.subheader("💡 Example Questions")

example_questions = [
    "How many annual leave days are employees entitled to?",
    "Can employees work remotely?",
    "What are the standard working hours?",
    "How many sick leave days are available?",
    "What is the resignation notice period?",
    "What is the employee salary?"
]

cols = st.columns(2)

for i, question in enumerate(
    example_questions
):

    with cols[i % 2]:

        if st.button(
            question,
            use_container_width=True
        ):

            st.session_state.question = (
                question
            )


# ---------------------------------------------------------
# Question Input
# ---------------------------------------------------------

st.subheader("💬 Ask HR Policy Assistant")

question = st.text_input(
    "Enter your question",
    value=st.session_state.get(
        "question",
        ""
    ),
    placeholder="e.g. What is the resignation notice period?"
)


# ---------------------------------------------------------
# Ask Question
# ---------------------------------------------------------

if st.button(
    "🔎 Ask Question",
    type="primary",
    use_container_width=True
):

    if not groq_api_key:

        st.warning(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()

    if "index" not in st.session_state:

        st.warning(
            "Please upload an HR Policy PDF first."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    with st.spinner(
        "Searching HR policy..."
    ):

        retrieved_chunks = retrieve_chunks(
            question,
            st.session_state.index,
            st.session_state.chunks,
            top_k=4
        )

    with st.spinner(
        "Generating answer..."
    ):

        try:

            answer = generate_answer(
                question,
                retrieved_chunks,
                groq_api_key
            )

        except Exception as e:

            st.error(
                f"Error while generating response: {str(e)}"
            )

            st.stop()

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    st.markdown("### 🤖 Answer")

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Retrieved Sources
    # -----------------------------------------------------

    with st.expander(
        "📚 View Retrieved Policy Sections"
    ):

        for i, item in enumerate(
            retrieved_chunks
        ):

            st.markdown(
                f"**Section {i + 1} — "
                f"Similarity: {item['score']:.3f}**"
            )

            st.write(
                item["text"]
            )

            st.markdown("---")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "HR Policy Assistant • RAG + FAISS + "
    "Sentence Transformers + Groq"
)

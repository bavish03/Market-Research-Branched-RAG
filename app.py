import streamlit as st
import chromadb
from router import route_question
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<style>

/* Main Page */
.stApp {
    background-color: #FFFFFF;
}

/* Main Heading */
.main-title {
    text-align:center;
    font-size:64px;
    font-family:Georgia, serif;
    color:#000000;
    font-weight:700;
}

/* Gold Highlight */
.highlight {
    color: #C8A05A;
    font-style: italic;
}

/* Cards */
.card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid #EAEAEA;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #111111;
    line-height: 1.7;
}

/* Input Box */
.stTextInput input {
    border-radius: 12px;
    border: 1px solid #D8D8D8;
    background-color: white;
    color: black;
}

/* Search Button */
.stButton > button {
    background-color: #C8A05A;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 25px;
    font-weight: 600;
}

/* Hover Effect */
.stButton > button:hover {
    background-color: #B38E4F;
    color: white;
}

/* Subheaders */
h2, h3 {
    color: #111111;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FAFAFA;
}

/* Branch Labels */
.branch-box {
    background-color: #FFF8EC;
    color: #C8A05A;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<h1 class='main-title'>
Market Research
<span class='highlight'>RAG</span>
</h1>

<p style='text-align:center;
          color:#666666;
          font-size:18px;
          margin-bottom:40px;'>

AI-Powered Market Intelligence Platform

</p>
""", unsafe_allow_html=True)

st.markdown(
"""
<hr style="
border:1px solid #E5E5E5;
margin-bottom:30px;">
""", unsafe_allow_html=True)

# -------------------------
# Cached Models
# -------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return OllamaLLM(model="phi3:mini")

model = load_embedding_model()
llm = load_llm()

# -------------------------
# Connect ChromaDB
# -------------------------

@st.cache_resource
def load_collections():
    client = chromadb.PersistentClient(path="./chroma_db")

    return (client.get_collection("companies"), client.get_collection("industry"), client.get_collection("news"))
company_collection, industry_collection, news_collection = load_collections()

with st.sidebar:
    st.markdown("## Research Controls")
    st.metric("Companies", company_collection.count())
    st.metric("Industry", industry_collection.count())
    st.metric("News", news_collection.count())

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Streamlit UI
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a market research question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    query_embedding = model.encode(question).tolist()
    routes = route_question(question)
    with st.sidebar:
        st.markdown("### Active Branches")
        for route in routes:
            st.success(route.upper())
    all_docs = []
    if "companies" in routes:
        results = company_collection.query(query_embeddings=[query_embedding], n_results=2)
        all_docs.extend(results["documents"][0])
    if "industry" in routes:
        results = industry_collection.query(query_embeddings=[query_embedding], n_results=2)
        all_docs.extend(results["documents"][0])
    if "news" in routes:
        results = news_collection.query(query_embeddings=[query_embedding], n_results=2)
        all_docs.extend(results["documents"][0])

    if len(all_docs) == 0:
        st.warning("No relevant documents found.")
        st.stop()
    context = "\n\n".join(all_docs)
    history = ""
    for msg in st.session_state.messages[-6:]:
        history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )
    prompt = f"""
You are a professional market research analyst.

Conversation History:
{history}

Retrieved Context:
{context}

Current Question:
{question}

Provide:

1. Executive Summary
2. Key Findings
3. Market Implications
4. Strategic Recommendations

Only use information from the retrieved context.
If information is missing, clearly state that.
"""
    with st.spinner("Analyzing market data..."):
        answer = llm.invoke(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    with st.expander("📄 View Retrieved Sources"):
        st.write(f"Retrieved {len(all_docs)} source chunks")
        for doc in all_docs:
            st.markdown(
                f"""
                <div class='card'>
                    {doc[:1000]}
                </div>
                """,
                unsafe_allow_html=True
            )
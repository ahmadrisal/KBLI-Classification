import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

# Pinecone & LangChain imports
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel

# Load environment variables
load_dotenv()
openai_api_key   = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    st.error("🔑 OPENAI_API_KEY belum diset. Tambahkan di .env atau environment.")
    st.stop()
if not pinecone_api_key:
    st.error("🔑 PINECONE_API_KEY belum diset. Tambahkan di .env atau environment.")
    st.stop()

# Chat history file
HISTORY_FILE = os.path.join("chat_history", "history.json")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

def load_history():
    """
    Baca file history.json dan normalisasi setiap pesan
    sehingga selalu ada key 'role' (user/assistant).
    """
    raw = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except:
            return []
    out = []
    for m in raw:
        # jika ada 'type', konversi ke 'role'
        if "type" in m:
            if m["type"] == "human":
                role = "user"
            elif m["type"] == "ai":
                role = "assistant"
            else:
                role = m["type"]
        else:
            # jika sudah pakai 'role', gunakan langsung
            role = m.get("role", "user")
        out.append({
            "role": role,
            "content": m.get("content", "")
        })
    return out

def save_history(msgs):
    """
    Simpan list pesan dengan key 'role' dan 'content'.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

def reset_history():
    save_history([])
    st.session_state.messages = [{"role":"assistant","content":"Tanyakan apa saja tentang KBLI"}]

class PineconeIndexManager:
    def __init__(self, embed_model: Embeddings, index_name: str):
        self.embed_model = embed_model
        self.index_name  = index_name
        self.api_key     = "pcsk_4mXCMJ_ANjVcgHrC2gMq7Gs68BRPbFSdwqW6JU1tgwxaUg4VgAUF7aw43iDbYfC5u8p6HY"
        # self.region      = config.get("pinecone_region", os.environ.get("PINECONE_ENVIRONMENT"))
        # init client
        self.pc = Pinecone(api_key=self.api_key)
        # ensure index exists
        self._create_index_if_not_exists()
        # open index handle
        self.index = self.pc.Index(self.index_name)

    def _create_index_if_not_exists(self):
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.region or "us-east-1")
            )
            # tunggu hingga index siap
            while True:
                desc = self.pc.describe_index(self.index_name)
                if desc.status.get("ready", False):
                    break
                time.sleep(1)

    def load_vector_store(self) -> VectorStore:
        # cukup buka kembali wrapper
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embed_model
        )
        print(f"Loaded PineconeVectorStore on index '{self.index_name}'.")
        return self.vector_store


# Inisialisasi chain
def init_chain():
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    manager     = PineconeIndexManager(embed_model=embed_model, index_name="kbli2020-index")
    vectordb    = manager.load_vector_store()
    llm         = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)

    system_instructions = """
Answer with:
- **Kode:** <kode KBLI>
- **Nama:** <classification name>
- **Deskripsi:** <detailed description>
If the question is unrelated to KBLI, say you cannot answer.
""".strip()

    discern_prompt = PromptTemplate(
        input_variables=["chat_history","question"],
        template=(
            "Rephrase follow‑up question as standalone.\n"
            "Conversation:\n{chat_history}\n"
            "Follow‑up: {question}\n"
            "Standalone:"
        )
    )

    combine_prompt = PromptTemplate(
        input_variables=["context","question"],
        template=f"""{system_instructions}

Context:
{{context}}

Question: {{question}}

Answer:"""
    )

    retriever = get_kbli_retriever(vectordb, llm, embed_model, top_k=3)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=discern_prompt,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        chain_type="stuff",
        return_source_documents=True,
    )

# Builder retriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser, get_query_constructor_prompt
)

def get_kbli_retriever(
    vector_store: VectorStore,
    llm_model: BaseLanguageModel,
    embed_model: Embeddings,
    top_k: int = 3
):
    retriever_sim = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":top_k})
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":top_k, "fetch_k":top_k*3})

    pinecone_translator = PineconeTranslator()
    metadata_info = [
        AttributeInfo(name="kategori", description="Primary KBLI category", type="string"),
        AttributeInfo(name="digit",    description="Number of digits in code", type="string"),
        AttributeInfo(name="kode",     description="Full KBLI code", type="string"),
        AttributeInfo(name="judul",    description="KBLI classification title", type="string"),
    ]
    schema_prompt = PromptTemplate.from_template(
        "<< Structured Request Schema >>\nUse a JSON snippet with 'query' and 'filter'."
    )
    prompt = get_query_constructor_prompt(
        document_contents="metadata and content",
        attribute_info=metadata_info,
        schema_prompt=schema_prompt,
        allowed_comparators=pinecone_translator.allowed_comparators,
        allowed_operators=pinecone_translator.allowed_operators
    )
    output_parser    = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm_model | output_parser

    retriever_self = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        search_type="mmr",
        search_kwargs={"k":top_k, "lambda_mult":0.85, "fetch_k":40}
    )

    merged = MergerRetriever(retrievers=[retriever_mmr, retriever_sim, retriever_self])
    redundancy_filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    compressor        = DocumentCompressorPipeline(transformers=[redundancy_filter])
    return ContextualCompressionRetriever(
        base_retriever=merged,
        base_compressor=compressor
    )

# Streamlit App
st.set_page_config(page_title="KBLI Chatbot", page_icon="📚", layout="centered")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = init_chain()
if "messages" not in st.session_state:
    st.session_state.messages = load_history() or [{"role":"assistant","content":"Tanyakan apa saja tentang KBLI"}]

st.title("KBLI Chatbot 🤖📚")

# User Input
if prompt := st.chat_input("Pertanyaan Anda tentang KBLI ..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    save_history(st.session_state.messages)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Generate and display response
# Generate and display response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    history_msgs = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))

    # <-- Tambahkan spinner di sini -->
    with st.spinner("Memproses jawaban…"):
        result = st.session_state.rag_chain.invoke({
            "question": st.session_state.messages[-1]["content"],
            "chat_history": history_msgs
        })

    answer = result["answer"]
    st.session_state.messages.append({"role":"assistant","content":answer})
    save_history(st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(answer)


# Reset button
if st.button("Reset"):
    reset_history()
    st.rerun()
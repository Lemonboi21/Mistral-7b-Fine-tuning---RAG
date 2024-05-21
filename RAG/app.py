
import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader , CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from sentence_transformers import SentenceTransformerEmbeddings 

from langchain_community.llms import LlamaCpp
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


def init_page() -> None:
    st.set_page_config(
        page_title="IT COMP Q&A Chatbot Assistant", 
        layout="wide"
    )

    st.header("IT COMP Q&A Chatbot Assistant")

    st.sidebar.title("Options")

def select_llm() -> LlamaCpp:
    return LlamaCpp(
        model_path="..\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        verbose=True,
    )

def load_embed_model():
    embed_model_Path = "..\models\sentence-transformers/all-mpnet-base-v2"

    embed_model = HuggingFaceEmbeddings(model_name=embed_model_Path)
    return embed_model 

def load_docs(directory):
    loader = CSVLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def create_db(docs , embed_model):
    persist_directory = "chroma_db"
    db = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=persist_directory
    )
    return db

def load_db():
    persist_directory = "chroma_db"
    db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
    return db

def load_chain(llm):
    return load_qa_chain(llm, chain_type="stuff")

def init_chatbot() -> None:
    llm = select_llm()
    embed_model = load_embed_model()
    documents = load_docs("../Data/mini_data.csv")
    docs = split_docs(documents)
    db = create_db(docs , embed_model)
    chain = load_chain(llm)
    return llm, db, chain

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear conversation" , key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="you are an AI assistant that can helps answering questions accurately provided by the context "
            ),
        ]

def get_answer(llm: LlamaCpp, db , chain , prompt: str) -> str:
    similar_docs = db.similarity_search(prompt, k=2)
    answer = chain.run(input_documents=similar_docs, question=prompt)
    return answer









def main() -> None:
    init_page()
    init_messages()
    llm, db, chain = init_chatbot()


    if st.chat_input("input your question here", key="input"):

        

        prompt = st.session_state.input

        with st.chat_message("user"):
            st.write(prompt)


        with st.spinner("Thinking..."):
            answer = get_answer(llm,db,chain, prompt)
            print(f"Answer: {answer}")

            message = st.chat_message("assistant")
            message.write(answer)

        #st.session_state.messages.append(SystemMessage(content=prompt))
        #st.session_state.messages.append(SystemMessage(content=answer))

   


if __name__ == "__main__":
    main()

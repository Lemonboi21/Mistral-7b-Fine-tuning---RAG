
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

from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

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
    embed_model_Path = "..\models\sentence-transformers/all-MiniLM-L6-v2"

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

def create_retriever(docs , embed_model):

    vector = FAISS.from_documents(docs, embed_model),
    print('*********************************************************************')
    print (vector)
    print('*********************************************************************')
    retriever = vector[0].as_retriever()
    
    return retriever


def load_chain(llm , retriever):
    
    
    prompt = ChatPromptTemplate.from_template("""You are a AI assistant for a company called 'IT Comp'. 
                                                your job is to answer questions you get using the context provided,no more and no less.
                                                if the context is irrelevant to the question you should ignore it.
                                            the answers should be proffesional and informative.
                                            you should not answer questions that are not related to the company.
                                            if you don't know the answer you should say so.

    <context>
    {context}
    </context>

    Question: {input}""")


    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def init_chatbot() -> None:
    llm = select_llm()
    embed_model = load_embed_model()
    documents = load_docs("../Data/mini_data.csv")
    #docs = split_docs(documents)
    retriever = create_retriever(documents , embed_model)
    chain = load_chain(llm , retriever)
    return llm, retriever, chain


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear conversation" , key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="you are an AI assistant that can helps answering questions accurately provided by the context "
            ),
        ]

def get_answer(llm: LlamaCpp, retriever , chain , prompt: str) -> str:
    answer = chain.invoke({"input": prompt})
    answer = answer["answer"]
    return answer









def main() -> None:
    init_page()
    init_messages()
    llm, retriever, chain = init_chatbot()

    


    if st.chat_input("input your question here", key="input"):

        

        prompt = st.session_state.input

        with st.chat_message("user"):
            st.write(prompt)


        with st.spinner("Thinking..."):
            answer = get_answer(llm,retriever,chain, prompt)
            print(f"{answer}")

            message = st.chat_message("assistant")
            message.write(answer)

        #st.session_state.messages.append(SystemMessage(content=prompt))
        #st.session_state.messages.append(SystemMessage(content=answer))

   


if __name__ == "__main__":
    main()

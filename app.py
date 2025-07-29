import json
import os
import boto3
import sys
import streamlit as st

##Using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

##Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

##LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

##Bedrock Clients
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=bedrock,region_name='us-east-1')

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # in our testing Character split works better wih this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vectoer_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

## LLM modle
def get_claude_llm():
    # instantiate the llm with bedrock
    llm = Bedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens":512})
    return llm

def get_nova_micro_llm():
    # instantiate the llm with bedrock
    llm = Bedrock(
        model_id="aamazon.nova-micro-v1:0",
        client=bedrock,
        model_kwargs={"maxTokens":512})
    return llm

prompt_template = """Human: Use the following pieces of context to provide a concise answer in 200 words to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
<context>
{context}
</context>
Question: {question}
Assistant:"""

def get_prompt():
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            serach_type="similarity", serach_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":get_prompt}
    )
    answer = qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Amazon Bedrock")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Create or update vectore store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vectoer_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


    if st.button("Nova Micro Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_nova_micro_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")       

main()
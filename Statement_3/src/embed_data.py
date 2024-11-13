"""
Perform inference/prediction on the test dataset and to prepare the submission file
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
#from pypdf import PdfReader #PDF Reading
import torch

#Import statements Langchain
from langchain.storage import InMemoryStore
from langchain.vectorstores import chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFacePipeline
# from langchain_huggingface import ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path 

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever

#Global Constants
DATA_PATH = r"..\..\masds002" #Path to raw data

def main():
    #data = os.path.join(DATA_PATH,"DS2-assessment-Simulated-Employee-Feedback.xslx")
    #print(pd.read_csv(data).head())
    dir_files = os.listdir(DATA_PATH)
    print(dir_files)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=hf_embeddings
    )

    docs = []
    loaders = []

    for file in dir_files:
        if file.startswith("DS3"):
            full_file_path = os.path.join(DATA_PATH,f"{file}")
            print(full_file_path)
            loaders.append(PyPDFLoader(full_file_path))
            perform_ocr(full_file_path,docs,loaders)

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600)

    store = InMemoryStore()
    parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
    )

    parent_retriever.add_documents(docs, ids=None)

    model_retriever = OllamaLLM(model="llama3.2",temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=parent_retriever, llm=model_retriever
    )
    # Set logging for the queries
    import logging

    # Set up logging to see your queries
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    # question = "what are the differences between the Swiss and UK acts?"
    # print(retriever_from_llm.invoke(question))
   
    model = OllamaLLM(model="llama3.2", temperature=0.3)

    system_prompt = (
    """ You are an assistant for question-answering tasks.
        Use only the following pieces of retrieved context to answer the question. If you do not know the answer, say that you don't know. 
        Keep the answer descriptive and verbose using information provided by the retrieved context to answer the question.

        \n \n
        {context}
    """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

    response = rag_chain.invoke({"input": "what are the differences between the Swiss and UK acts"})
    #print(response["answer"])
    #print(response["context"])
    source_dict = {}
    for doc in response["context"]:
        full_path = doc.metadata["source"]
        stem_path = Path(full_path).stem
        page = doc.metadata["page"]
        if stem_path not in source_dict:
            source_dict[stem_path] = set()
        source_dict[stem_path].add(page)
    generated_output = str(response["answer"])
    generated_output += f"\n\nSources in Document and Page Numbers format:"
    for source, page_numbers in source_dict.items():
        #page_numbers = str(list)
        generated_output += f"\n{source}: {page_numbers} "
    
    print(generated_output)
        
    pass

def perform_ocr(full_file_path,docs,loaders):

    for loader in loaders:
        docs.extend(loader.load())

    pass




if __name__ == "__main__":
    main()
    




 #print("This is the answer document",sub_docs)


    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id
    # )
    # device = 0 if torch.cuda.is_available() else -1
    # print(device,"this is the device")
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, top_k=50, temperature=0.1,device=device)
    # llm = HuggingFacePipeline(pipeline=pipe)
    # # print(llm.invoke("Hugging Face is"))

    # system_prompt = (
    # "You are an assistant for question-answering tasks. "
    # "Use only the following pieces of retrieved context to answer "
    # "the question. If you don't know the answer, say that you "
    # "don't know. Use three sentences maximum and keep the "
    # "answer concise."
    # "\n\n"
    # "{context}"
    # )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{input}"),
    #     ]
    # )


    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # response = rag_chain.invoke({"input": "What is the UK Act"})
    # print(response["answer"])
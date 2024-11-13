"""
Perform inference/prediction on the test dataset and to prepare the submission file
"""

#Import statements
import os #os file paths
from pathlib import Path #Manipulate file path names

#Import statements Langchain
from langchain.storage import InMemoryStore #Storage
from langchain_text_splitters import RecursiveCharacterTextSplitter #Chunking
from langchain.retrievers import ParentDocumentRetriever #RAG
from langchain_huggingface.embeddings import HuggingFaceEmbeddings #Embeddings
from langchain_chroma import Chroma #Vector Database
from langchain_community.document_loaders import PyPDFLoader #OCR for PDF
from langchain.chains.retrieval import create_retrieval_chain #RAG Chain
from langchain.chains.combine_documents import create_stuff_documents_chain #Retrieval Part of Chain
from langchain_core.prompts import ChatPromptTemplate #Chat Template for QA
from langchain_ollama.llms import OllamaLLM #LLM
from langchain.retrievers.multi_query import MultiQueryRetriever #RAG

#Global Constants
DATA_PATH = r"..\..\masds002" #Path to raw data/documents

def main():
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
    



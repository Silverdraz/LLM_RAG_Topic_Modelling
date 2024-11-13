"""
Extract the text from the relevant documents (e.g. PDF files), followed by embed these chunks 
"""

#Import statements
import os #os file paths

#Import statements Langchain
from langchain.storage import InMemoryStore #Storage
from langchain_text_splitters import RecursiveCharacterTextSplitter #Chunking
from langchain.retrievers import ParentDocumentRetriever #RAG
from langchain_huggingface.embeddings import HuggingFaceEmbeddings #Embeddings
from langchain_chroma import Chroma #Vector Database
from langchain_community.document_loaders import PyPDFLoader #OCR for PDF
from langchain_ollama.llms import OllamaLLM #LLM
from langchain.retrievers.multi_query import MultiQueryRetriever #RAG
from langchain.prompts import PromptTemplate #Customising Prompt Templating

#Import other modules in src
import custom_prompts #prompt modules

#Global Constants
DATA_PATH = r"..\..\masds002" #Path to raw data/documents
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" #Embedding model string
LLM_MODEL = "llama3.2" #LLM model


def docs_adder(docs,loaders):
    """Helper function to add the OCRed text into the docs list for create_embedding_model_function

    """   
    #Iterate through every doc that is pypdfloaded
    for loader in loaders:
        docs.extend(loader.load())

def create_embedding_model():
    """Create embedding model to embed the documents

    Returns:
        hf_embeddings: loaded hugging face embedding model to embed the documents/text
    """    
    #Create the parameters to load embedding model
    model_name = EMBEDDING_MODEL
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
    )
    return hf_embeddings

def perform_ocr():
    """ OCR - Return the text from the documents that are read

    Returns:
        docs: List of text from documents that are read. E.g. [doc1text, doc2text, ...]
    """    
    #Create the parameters to load embedding model
    dir_files = os.listdir(DATA_PATH)

    #Create docs to store OCRed documents using PyPDF Loaded documents
    docs = []
    loaders = []

    #Iterate through every dcoument
    for file in dir_files:
        #Filter only for the relevant documents
        if file.startswith("DS3"):
            full_file_path = os.path.join(DATA_PATH,f"{file}")
            #Use the loader to read the text
            loaders.append(PyPDFLoader(full_file_path))
            docs_adder(docs,loaders)
    return docs

def create_parent_retriever(hf_embeddings,docs):
    """ Create parent retriever variable for RAG by first chunking the text in parent and child chunks, followed by
    embedding the chunks. The embedded chunks would then be uploaded into the Chroma vector store

    Args:
        hf_embeddings: Hugging Face embedding model
        docs: List containing both the x_train dataframe and x_test dataframe

    Returns:
        parent_retriever: Parent Retriever to retrieve the relevant documents with more context (above or below) chunks
    """    
    #Create the chroma vector store and set the embeddings
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=hf_embeddings
    )


    # Chunking - This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)

    # Chunking - This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600)

    store = InMemoryStore()
    #Create the parent document retriever
    parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
    )

    #Ingest the relevant documents
    parent_retriever.add_documents(docs, ids=None)
    return parent_retriever

def create_multiqueries_retriever(parent_retriever):
    """ Create multi queries retriever by generating more queries using the llm

    Args:
        parent_retriever: Parent Retriever to retrieve the relevant documents with more context (above or below) chunks

    Returns:
        retriever_from_llm: retriever that generates more queries using LLM and then followed by retrieval from vector store
    """    

    #Create the LLM model used for the multi query retriever
    model_retriever = OllamaLLM(model=LLM_MODEL,temperature=0)

    #Custom prompt for multi query retriever
    PROMPT = PromptTemplate(
        template=custom_prompts.MULTI_QUERY_RETRIVAL_PROMPT, input_variables=["question"]
    )

    #Create the multi query retriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=parent_retriever, llm=model_retriever, prompt=PROMPT
    )
    return retriever_from_llm




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
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models, AsyncQdrantClient
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from storage.prompt_store.prompts import N3Prompts
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext, load_index_from_storage

import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from utils.file_loader import load_from_config

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document, load_index_from_storage
import tiktoken
from llama_index.core import ServiceContext
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings


from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import List, Tuple, Any
from pydantic import BaseModel, Field



public_data = load_from_config("scraped_data")
n3p = N3Prompts()



## 04. company abtract

token_counter = TokenCountingHandler(
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002").encode,
    verbose=True
)
callback_manager = CallbackManager([token_counter])
service_context = ServiceContext.from_defaults(callback_manager = callback_manager)


# prepare data for qdrant

iter_dict = lambda x:  ",\n".join([f"{k} : {v}"  for k,v in x.items() if v!=''])

def dict2str(i):
   pi = iter_dict(public_data[i]["poslovna.hr"]["poslovna_intro"])
   po = iter_dict(public_data[i]["poslovna.hr"]["poslovna_osnovno"])
   ddgo = iter_dict(public_data[i]["ddgo"])
   abs =  public_data[i]["poslovna.hr"]["abstract"]

   abs_2 ="" if abs=="" else f"\n\nDokument: \n\n{abs}"

   return f"{pi}\\n{po}\n\n{ddgo}{abs_2}"

def load_data_4_qdrant():
    FILTER_OF_RELEVANT_DOCS = """ i ==public_data[i]['poslovna.hr']['poslovna_intro']['OIB']  or public_data[i]['NAZIV'] ==public_data[i]['poslovna.hr']['poslovna_intro']['Naziv_subjekta'] or public_data[i]['NAZIV'] ==public_data[i]['poslovna.hr']['poslovna_intro']['Podnaslov_subjekta'] or public_data[i]['poslovna.hr']['poslovna_intro']['Podnaslov_subjekta']=='' """
    
    list_of_all = [{"oib": i, "naziv":public_data[i]["NAZIV"], "doc": dict2str(i)} for i in list(public_data.keys()) if eval(FILTER_OF_RELEVANT_DOCS)]



    list_of_doc_oibs = [i["oib"] for i in list_of_all]
    list_of_names = [i["naziv"] for i in list_of_all]
    list_of_docs= [Document(text=i["doc"]) for i in list_of_all]

    return list_of_doc_oibs, list_of_names, list_of_docs

 # simple

poslovna_documents = ["OIB: {} -> {}".format(i,public_data[i]["poslovna.hr"]["abstract"]) for i in list(public_data.keys()) if public_data[i]["poslovna.hr"]["abstract"]!='']


def load_create_public_data_simple_store():
    vdb_output_cid_abstract = "./storage/cache_2/company_abstract/"
    #cid_path = "data/cids/"

    #token_counter = TokenCountingHandler(
    #    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002").encode,
    #    verbose=True
    #)

    #callback_manager = CallbackManager([token_counter])

    #service_context = ServiceContext.from_defaults(callback_manager = callback_manager)


    try:
        # loading from disk
        storage_context_abstract = StorageContext.from_defaults(persist_dir=vdb_output_cid_abstract)
        index = load_index_from_storage(storage_context_abstract)
        print('loading from disk')
    except:
        # documents_abstract = [Document(text=t) for t in poslovna_documents] #SimpleDirectoryReader(cid_path).load_data()
        # generating embeddings - default model openai ada
        index = VectorStoreIndex.from_documents(
            []#,
            #storage_context=storage_context,
        )
        pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])

        list_of_doc_oibs, list_of_names, list_of_docs = load_data_4_qdrant()


        for oib, name, doc in zip(list_of_doc_oibs, list_of_names, list_of_docs):
            nodes = pipeline.run(documents=[doc])
            for node in nodes:
                node.metadata = {"oib" : oib}
                node.metadata = {"naziv" : name}
            index.insert_nodes(nodes)

        # index_abstract = VectorStoreIndex.from_documents(documents_abstract, service_context=service_context ,show_progress=True)




        index.storage_context.persist(persist_dir=vdb_output_cid_abstract)
        print('persisting to disk')

    from llama_index.core.prompts import PromptTemplate

    qa_prompt_abstract = PromptTemplate(
        n3p.public_data_prompt

    )

    abstract_qa_template = qa_prompt_abstract


    c_abstrract_qe = index.as_query_engine(text_qa_template=abstract_qa_template)

    c_abstract_tool = QueryEngineTool(query_engine=c_abstrract_qe, metadata=ToolMetadata(
            name="public_company_data",
            description="this tool finds company public information using company name"
        ))

    return c_abstract_tool

# qdrant


qdrant_store_vdb = "storage/qdrant_cache/q_public_data_2"


def load_create_public_data_qdrant_store():

    try:

        # Check if the path exists
        if not os.path.exists(qdrant_store_vdb):
            raise FileNotFoundError("The specified path does not exist.")

        client = QdrantClient(path=qdrant_store_vdb)  # replace with your Qdrant server details

        # Define the collection name where the vectors are stored
        collection_name = "public_company_data_q"
        # Create the QdrantVectorStore instance
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("loading from disk")
    except:

        client = QdrantClient(path=qdrant_store_vdb) # QdrantClient(location=":memory:")
        client.create_collection(
        collection_name="public_company_data_q",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
        vector_store = QdrantVectorStore(client=client, collection_name="public_company_data_q")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )
        pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])

        list_of_doc_oibs, list_of_names, list_of_docs = load_data_4_qdrant()


        for oib, name, doc in zip(list_of_doc_oibs, list_of_names, list_of_docs):
            nodes = pipeline.run(documents=[doc])
            for node in nodes:
                node.metadata = {"oib" : oib}
                node.metadata = {"naziv" : name}
            index.insert_nodes(nodes)

    qa_prompt_abstract = PromptTemplate(
        n3p.public_data_prompt

    )

    qe = index.as_query_engine(qa_template= qa_prompt_abstract)

    c_abstract_tool = QueryEngineTool(query_engine=qe, metadata=ToolMetadata(
            name="public_company_data",
            description="this tool finds company public information using company name"
        ))


    


    
    return c_abstract_tool


def pick_abstract_tool(qdrant=False):
    if qdrant:
        tool = load_create_public_data_qdrant_store()
    else:
        tool = load_create_public_data_simple_store()
    print (tool)
    
    return tool
import numpy as np
import pandas as pd
import pandas as pd
from duckduckgo_search import DDGS
import requests
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import json
import pickle
from utils.file_loader import load_from_config
from dotenv import  load_dotenv
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import tiktoken
from llama_index.core import ServiceContext
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import tiktoken
from llama_index.core import ServiceContext
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings


from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from openai import OpenAI
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.prompts import PromptTemplate

from storage.prompt_store.prompts import N3Prompts
import logging
from utils.db_tools_async import internal_data_tool_360, engine, Base, MobileCustomerBase, generate_context_for_internal_data_async, id_sync
from sqlalchemy.orm import sessionmaker
from utils.public_data_tools import pick_abstract_tool
import asyncio

Session = sessionmaker(bind=engine)
session = Session()
#Base.metadata.create_all(engine)
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Run the table creation
#asyncio.run(create_tables())


n3p = N3Prompts()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("moj loger")


#Settings.llm = OpenAI(temperature=0.2, model="gpt-4o")

load_dotenv()

logger.warning(Settings.llm.model)
Settings.llm.model = "gpt-4o"
Settings.llm.temperature=0.8
logger.warning(Settings.llm.model)


public_data = load_from_config("scraped_data")
company_ids = pd.read_csv("data/sample_cb.dsv", sep="|", encoding="cp1250")
oib_list = [f"oib (vat_id): {i[0]}, naziv: {i[1]}" for i in zip(company_ids["OIB"],company_ids["NAZIV"])]

cid_file_path = "data/cids/company_ids.txt"

subscriber_level_df_final = pd.read_csv("data/synthetic_data_cb_w_oib.csv")


#with open(cid_file_path,"w") as file:
#    file.write("\n".join(oib_list))
#file.close()

# oib list for get company name based on company oib

oib2name = {str(i[0]):{"oib": i[0], "naziv": i[1]} for i in zip(company_ids["OIB"],company_ids["NAZIV"])}

def oib2name_func(oib:str):
    """
    input is numeric string oib
    Finds company name based on oib input.
    Use this tool when user inputs oib and you need company name
    output is dictionary with oib and name of the company
    """
    oib2name_ = oib2name

    return str(oib2name_[oib])


tool_oib2name = FunctionTool.from_defaults(oib2name_func, 
name="oib2name",
description= """
    Finds company name based on oib input.
    Use this tool when user inputs oib and you need company name
    output is dictionary with oib and name of the company
    """,
return_direct=True)



# ===========================================
# 01 find company name tool

vdb_output_cid = "./storage/cache/company_ids/"
cid_path = "data/cids/"

token_counter = TokenCountingHandler(
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002").encode,
    verbose=True
)



callback_manager = CallbackManager([token_counter])

service_context = ServiceContext.from_defaults(callback_manager = callback_manager)


try:
    # loading from disk
    storage_context = StorageContext.from_defaults(persist_dir=vdb_output_cid)
    index = load_index_from_storage(storage_context)
    print('loading from disk')
except:
    documents = [Document(text=t) for t in oib_list] #SimpleDirectoryReader(cid_path).load_data()
    # generating embeddings - default model openai ada
    index = VectorStoreIndex.from_documents(documents, service_context=service_context) # ,show_progress=True)
    index.storage_context.persist(persist_dir=vdb_output_cid)
    print('persisting to disk')

print(token_counter.total_embedding_token_count)


qa_prompt = PromptTemplate(
    n3p.cid_qe_prompt
)

cid_qe = index.as_query_engine(text_qa_template=qa_prompt)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

cid_qe_tool = QueryEngineTool(query_engine=cid_qe, metadata=ToolMetadata(
        name="company_name_and_id_finder",
        description="this tool finds company name and oib based on user input"
    ))


# 02 nlsql query engine tool

# from llama_index.llms.openai import OpenAI

# llm = OpenAI(model="gpt-4o")

from sqlalchemy import create_engine
engine = create_engine("sqlite+pysqlite:///:memory:")
subscriber_level_df_final.to_sql(
  "mobile_customer_base",
  engine
)

tables = ["mobile_customer_base"]

from llama_index.core import SQLDatabase

sql_database = SQLDatabase(
    engine=engine,
    include_tables=tables
)

from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine


examples = [
    {"input":"what customer has the most subscribers","output":"select oib,count(subscriber_id) from mobile_customer_base group by oib sort by count(subscriber_id) desc LIMIT 1"},
    {"input":"what oib has the most subscribers that contract expiration date is less than 3 months","output":"select oib,count(subscriber_id) from mobile_customer_base where fees_to_charge <3 group by oib sort by count(subscriber_id) desc LIMIT 1"},
     {"input":"koji korisnik ima najviše pretplatnika kojima ističe ugovor za manje od 3 mjeseca","output":"select oib,count(subscriber_id) from mobile_customer_base where fees_to_charge <3 group by oib sort by count(subscriber_id) desc LIMIT 1"},
   
    
    ]

#prompt_temp = NLSQLTableQueryEngine(sql_database=sql_database,
#    tables=tables).get_prompts()["sql_retriever:text_to_sql_prompt"]
#prompt_temp.template=prompt_change


sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=tables,
    kwargs=examples,
    #text_to_sql_prompt=prompt_temp
    
    
)


DESCRIPTION = n3p.sql_qe_template["query_tool_engine_desc"]
"""
Ovaj alat koristi se za prevođenje upita na prirodnom jeziku u SQL upite s ciljem dobivanja informacija o mobilnim korisnicima iz baze podataka. Ključne značajke i pravila korištenja su:

mobile_customer_base: Tablica koja sadrži podatke o mobilnim korisnicima.

"""
#n3p.sql_qe_template["query_tool_engine_desc"]

# You only search by oib, if customer doesn't provide oib number or result of your query is empty answer 'I don't have that info'




from llama_index.core.tools.query_engine import QueryEngineTool

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql-query",
    description=DESCRIPTION

)



#n3p.sql_qe_template["query_tool_custom_prompt"]
#sql_query_engine.get_prompts()['sql_retriever:text_to_sql_prompt'].template=prompt_change
# sql_query_engine._update_prompts


from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.agent.openai import OpenAIAgent


# tool_spec = DuckDuckGoSearchToolSpec()
# duck_duck_go_tool = tool_spec.to_tool_list()[1]

# agent = OpenAIAgent.from_tools([duck_duck_go_tool], verbose=True)

from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from openai import OpenAI


client = OpenAI()

def user_query(prompt):
    metadata = f"""Potraži vijesti o firmi {prompt} i sažmi ih u kratki sažetak od 5 rečenica. 
    Odogovor neka preferira informacije s portala kao što su jutarnji.hr, poslovni.hr, index.hr, dnevnik.hr, vecernji.hr i drugih hrvatski news portala. 
    Izbjegavaj iznositi podatke sa stranica koje sadrže osnovne informacije o firmi kao npr. fininfo ili poslovna.hr. 
    Ne navodi čime se tvrtka bavi, već se fokusiraj na aktivnosti, potražnju radnika, afere i slično.
    Nakon pretrage provjeri sadrže li razultati pretrage informacije o tvrtki {prompt}.
    One rezultate koji ne sadrže, izostavi iz finalnog odgovora.

    Osgovor neka bude prema primjeru:

    1. Tvrtka Primjer je otvorila novi ured prije mjesec dana (jutarnji.hr)
    2. Istraga otvorene nakon smunji u korupciju u tvrtki Primjer (index.hr)
    """

    return metadata

def search_web(query: str):
    """Potraži vijesti o firmi i sažmi ih u kratki sažetak od 5 rečenica. 
    Izbjegavaj iznositi podatke sa stranica koje sadrže osnovne informacije o firmi kao npr. fininfo ili poslovna.hr. 
    Vijesti neka imaju prednost u prikazu ispred osnovnih informacija o tvrtki.
    Nakon pretrage provjeri sadrže li razultati pretrage informacije o tvrtki.
    One rezultate koji ne sadrže, izostavi iz finalnog odgovora.

    Odgovor neka bude prema primjeru:

    1. Tvrtka Primjer je otvorila novi ured prije mjesec dana (jutarnji.hr)
    2. Istraga otvorene nakon smunji u korupciju u tvrtki Primjer (index.hr)
    """

    #search_prompt = user_query(query)

    #tool_spec = DuckDuckGoSearchToolSpec()
    #duck_duck_go_tool = tool_spec.to_tool_list()[1]

    #agent = OpenAIAgent.from_tools([duck_duck_go_tool], verbose=True, max_function_calls=2)

    #resp = agent.query(search_prompt)

    web_resp = DDGS().text(f"{query} u vijestima", max_results=5, region="hr-hr")

    

    user = n3p.get_news_search_prompt(web_resp=web_resp)
  
    system = n3p.news_search_prompt["system"]

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[system,user,user],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    

    return response.choices[0].message.content


web_search_tool = FunctionTool.from_defaults(fn=search_web, name="search_company_news", return_direct=True)


    #public_data.keys()

## 04. company abtract


c_abstract_tool = pick_abstract_tool()




from llama_index.core.tools import FunctionTool



async def sync_function_wrapper(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)



## 05. customer 360

async def customer_360_new(qry):

    """
    ovaj alat daje pregled firme od 360 stupnjeva, ulaz alata je u formatu biblioteke npr. {{'oib': '1111111111','naziv': 'Naziv firme'}}
    odgovor svakog od alata (public data, internal data, news) ne smije biti duži od 250 tokena
    ne prikazuj vcijesti sa stranica: poslovna.hr, infobiz.fina.hr, fininfo.hr

    odgovor je u formatu:

    ## Javni podaci
    {resp_public_data}

    ## Interni podaci
    {resp_id}

    ## {qry['naziv']} u vijestima
    {resp_news_data}


    """
    #print(qry)
    #qry_dv =f"find avg for arpa, voice usage, discount, count of subscribers and overshoot and group by tariff_model and filter by oib = {qry['oib']}"
    #qry_dv = n3p.get_sql_report_prompt(qry)
    
    # resp_id = sql_tool.call(qry_dv)
    # Wrap synchronous functions in async wrappers
    internal_data_task = asyncio.create_task(generate_context_for_internal_data_async(qry['oib']))
    news_data_task = asyncio.create_task(sync_function_wrapper(web_search_tool, qry['naziv']))
    
    # Call asynchronous function directly
    public_data_task = asyncio.create_task(sync_function_wrapper(c_abstract_tool, qry['naziv']))

    # Await all tasks to complete
    resp_id = await internal_data_task
    resp_news_data = await news_data_task
    resp_public_data = await public_data_task
    #resp_id = id_sync(qry)


    ## Interni podaci\n{resp_id}\n
    #result = f"""## Javni podaci\n{resp_public_data}\n## {qry['naziv']} u vijestima\n{resp_news_data}"""
    
    return resp_id, resp_news_data, resp_public_data

tool_360 = FunctionTool.from_defaults(customer_360_new, 
    name="customer_360_new", 
    #description="this tool gives customer 360 overview in Croatian language, input of tool is dict in form of e.g. {{'oib': '1111111111','naziv': 'Naziv firme'}}",
    return_direct=False,
    async_fn=customer_360_new
    
    )

# 06. build agenta

from llama_index.core.agent import FunctionCallingAgentWorker, ReActAgent

import inspect

agent_worker2 = FunctionCallingAgentWorker.from_tools(
    tools=[sql_tool, cid_qe_tool, tool_oib2name, c_abstract_tool, tool_360, web_search_tool],
    verbose=True,
    system_prompt=n3p.nnn_agent_nmhr
)


agent2 = agent_worker2.as_agent()
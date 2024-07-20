import pandas as pd
from storage.prompt_store.prompts import N3Prompts
from llama_index.core.tools import FunctionTool
from openai import OpenAI



client = OpenAI()
n3p = N3Prompts()
# subscriber_level_df_final = pd.read_csv("data/synthetic_data_cb_w_oib.csv")


from sqlalchemy import create_engine
# engine2 = create_engine("sqlite+pysqlite:///:memory:")
engine2 = create_engine("sqlite+pysqlite:///db//internal_database.db")
# subscriber_level_df_final.to_sql(
#  "mobile_customer_base",
#  engine2
#  )

tables = ["mobile_customer_base"]


from sqlalchemy import create_engine, func, select, case
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float

Base = declarative_base()

class MobileCustomerBase(Base):
    __tablename__ = tables[0]
    subscriber_id = Column(Integer, primary_key=True)
    oib = Column(String)
    tariff_model = Column(String)
    price_overshoot = Column(Float)
    voice_usage = Column(Float)
    roaming_usage = Column(Float)
    arpa_2024 = Column(Float)
    arpa_2023 = Column(Float)
    revenue_2023 = Column(Float)
    revenue_2024 = Column(Float)
    fees_to_charge =Column(Integer)

Session = sessionmaker(bind=engine2)
session = Session()

# Kreiranje upita koristeći SQLAlchemy
def generate_context_for_internal_data(oib:str):

    """
    use only for 360 tool
    run sql query and returns text results of db query

    based on the output agent should create internal data summary
    """

    total_connections_subquery = (
        session.query(func.count(MobileCustomerBase.subscriber_id))
        .filter(MobileCustomerBase.oib == oib)  # or MobileCustomerBase.oib.is_(None) if checking for NULLs
        .scalar_subquery()
    )

    # Create the main query to calculate the share for each tariff model
    query = (
        session.query(
            MobileCustomerBase.tariff_model,
            func.count(MobileCustomerBase.subscriber_id).label('total_connections'),
            func.avg(MobileCustomerBase.price_overshoot).label('avg_overshoot'),
            func.avg(MobileCustomerBase.voice_usage).label('avg_voice_usage'),
            func.avg(MobileCustomerBase.roaming_usage).label('avg_roaming_usage'),
            func.round(func.count(MobileCustomerBase.subscriber_id) / total_connections_subquery * 100.0,2).label('tariff_mix')
        )
        .filter(MobileCustomerBase.oib == oib)  # or MobileCustomerBase.oib.is_(None) if checking for NULLs
        .group_by(MobileCustomerBase.tariff_model)
    )

    # Izvršavanje upita i dobivanje rezultata
    results = query.all()

    # Define headers
    headers = [('tariff_model', 'total_connections', 'avg_overshoot', 'avg_voice_usage', 'avg_roaming_usage',"tariff_mix")]

    full_result = headers+results

    row_str_result = lambda rec:"\t".join(str(r) for r in rec)
    table_str_result = lambda row:"\n".join(row_str_result(r) for r in row)

    str_result = table_str_result(full_result)

    # financial data

    prihod_delta = (
        session.query(
            func.avg(MobileCustomerBase.arpa_2024).label('arpa_2024'),
            (func.round((func.avg(MobileCustomerBase.arpa_2024)/func.avg(MobileCustomerBase.arpa_2023))-1,2)*100).label("arpa_trend"),
            
            func.sum(MobileCustomerBase.revenue_2024).label('revenue_2024'),
            
            (func.round((func.sum(MobileCustomerBase.revenue_2024)/ func.sum(MobileCustomerBase.revenue_2023))-1,2)*100).label("revenue_trend"),
            func.count(MobileCustomerBase.subscriber_id).label('ukupan broj korisnika'),
            func.sum(case( 
            (MobileCustomerBase.fees_to_charge<7, 1),
            else_=0)).label('6_months_til_contract_expiry'),
            func.sum(case( 
            (MobileCustomerBase.fees_to_charge<4, 1),
            else_=0)).label('3_months_til_contract_expiry')
        ).filter(MobileCustomerBase.oib == oib)
       
    )
    
    # Izvršavanje upita i dobivanje rezultata
    rev_results = prihod_delta.all()

    # Define headers
    rev_headers = [('arpa_2024', 'arpa_trend', 'revenue_2024', 'revenue_trend',"broj priključaka u bazi",'broj priključaka kojima guvor ističe za 3 mjeseca','broj priključaka kojima ugovor ističe za 6 mjeseci')]

    rev_full_result = rev_headers+rev_results

    rev_result = table_str_result(rev_full_result)

    concant_result = "{}\n{}".format(str_result, rev_result)




    user = n3p.get_sql_360_prompt(concant_result)
  
    system = n3p.internal_360_id["system"]

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[system,user],
    temperature=0.2,
    max_tokens=800,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    

    return response.choices[0].message.content


internal_data_tool_360 = FunctionTool.from_defaults(fn=generate_context_for_internal_data, name="sql_internal_data_tool")
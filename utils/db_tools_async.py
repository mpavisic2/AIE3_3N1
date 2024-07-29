import pandas as pd
from storage.prompt_store.prompts import N3Prompts
from llama_index.core.tools import FunctionTool
from openai import OpenAI
import logging
import asyncio

# Importing libraries and modules required for various features in the script

client = OpenAI()  # Initialize OpenAI client for API interactions
n3p = N3Prompts()  # Instance of N3Prompts to manage prompt-related operations

# Importing database management tools from SQLAlchemy
from sqlalchemy import create_engine, func, select, case
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float

# Configure logging to help in debugging and tracking the flow and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()  # Base class for declarative class definitions

# Configuration for the database
DATABASE_URL = "sqlite+aiosqlite:///db//internal_database.db"
engine = create_async_engine(DATABASE_URL, echo=True)  # Asynchronous engine for database operations

# Create a session factory that will create new session objects when called
async_session = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# ORM model class for mobile customers
class MobileCustomerBase(Base):
    __tablename__ = "mobile_customer_base"  # Define the table name
    subscriber_id = Column(Integer, primary_key=True)  # Define a primary key column
    oib = Column(String)  # Other columns of various types
    tariff_model = Column(String)
    price_overshoot = Column(Float)
    voice_usage = Column(Float)
    roaming_usage = Column(Float)
    arpa_2024 = Column(Float)
    arpa_2023 = Column(Float)
    revenue_2023 = Column(Float)
    revenue_2024 = Column(Float)
    fees_to_charge = Column(Integer)

# Function to generate data context asynchronously
async def generate_context_for_internal_data_async(oib: str):
    try:
        async with async_session() as session:
            # Constructing a subquery for total connections based on 'oib'
            total_connections_subquery = (
                select(func.count(MobileCustomerBase.subscriber_id))
                .filter(MobileCustomerBase.oib == oib)
                .scalar_subquery()
            )

            # Main query to gather tariff model and its statistical usage data
            query = (
                select(
                    MobileCustomerBase.tariff_model,
                    func.count(MobileCustomerBase.subscriber_id).label('total_connections'),
                    func.avg(MobileCustomerBase.price_overshoot).label('avg_overshoot'),
                    func.avg(MobileCustomerBase.voice_usage).label('avg_voice_usage'),
                    func.avg(MobileCustomerBase.roaming_usage).label('avg_roaming_usage'),
                    (func.round(func.count(MobileCustomerBase.subscriber_id) / total_connections_subquery * 100.0, 2)).label('tariff_mix')
                )
                .filter(MobileCustomerBase.oib == oib)
                .group_by(MobileCustomerBase.tariff_model)
            )

            results = await session.execute(query)  # Execute the query asynchronously
            results = results.fetchall()  # Fetch all results

            # Formatting results for display or further processing
            headers = [('tariff_model', 'total_connections', 'avg_overshoot', 'avg_voice_usage', 'avg_roaming_usage', "tariff_mix")]
            full_result = headers + results
            row_str_result = lambda rec: "\t".join(str(r) for r in rec)
            table_str_result = lambda row: "\n".join(row_str_result(r) for r in row)
            str_result = table_str_result(full_result)

            # Additional revenue-related query
            prihod_delta = (
                select(
                    func.avg(MobileCustomerBase.arpa_2024).label('arpa_2024'),
                    (func.round((func.avg(MobileCustomerBase.arpa_2024) / func.avg(MobileCustomerBase.arpa_2023)) - 1, 2) * 100).label("arpa_trend"),
                    func.sum(MobileCustomerBase.revenue_2024).label('revenue_2024'),
                    (func.round((func.sum(MobileCustomerBase.revenue_2024) / func.sum(MobileCustomerBase.revenue_2023)) - 1, 2) * 100).label("revenue_trend"),
                    func.count(MobileCustomerBase.subscriber_id).label('ukupan broj korisnika'),
                    func.sum(case((MobileCustomerBase.fees_to_charge < 7, 1), else_=0)).label('6_months_til_contract_expiry'),
                    func.sum(case((MobileCustomerBase.fees_to_charge < 4, 1), else_=0)).label('3_months_til_contract_expiry')
                ).filter(MobileCustomerBase.oib == oib)
            )

            rev_results = await session.execute(prihod_delta)  # Execute revenue query
            rev_results = rev_results.fetchall()  # Fetch results

            rev_headers = [('arpa_2024', 'arpa_trend', 'revenue_2024', 'revenue_trend', "broj priključaka u bazi", 'broj priključaka kojima ugovor ističe za 3 mjeseca', 'broj priključaka kojima ugovor ističe za 6 mjeseci')]
            rev_full_result = rev_headers + rev_results
            rev_result = table_str_result(rev_full_result)

            concant_result = "{}\n{}".format(str_result, rev_result)  # Combine all results into one formatted string

        # Fetching and using prompts from n3p for generating a response using OpenAI's GPT-4 model
        user = n3p.get_sql_360_prompt(concant_result)
        system = n3p.internal_360_id["system"]
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[system, user],
                temperature=0.2,
                max_tokens=800,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

        return response.choices[0].message.content  # Return the generated response
    except Exception as e:
        logger.error(f"Error in generate_context_for_internal_data_async: {e}")
        return f"Error: {e}"

# Sync wrapper function to call asynchronous data generation function
def id_sync(qry):
    resp_id = asyncio.run(generate_context_for_internal_data_async(qry['oib']))  # Run the async function synchronously
    return resp_id 

# Create a FunctionTool object for better modularity and easier function calling
internal_data_tool_360 = FunctionTool.from_defaults(fn=generate_context_for_internal_data_async, name="sql_internal_data_tool", description="async alat koji generira interne podatke za 360 pogled, potrebnop pozvait await prilikom generianja odgvora", async_fn=generate_context_for_internal_data_async)

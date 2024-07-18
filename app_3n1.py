# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import chainlit as cl  # importing chainlit for our app
from dotenv import load_dotenv

from utils.nnn_tools import agent2

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():



    cl.user_session.set("chain", agent2, )





@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    resp = await chain.achat(message.content)

    msg = cl.Message(content=resp.response)

    #print(msg.content)
    await msg.send()

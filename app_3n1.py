# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import chainlit as cl  # importing chainlit for our app
from dotenv import load_dotenv

from utils.nnn_tools import agent2

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Započni razgovor",
            message="Ukratko se predstavi i ponudi pomoć",
            icon="/public/intro.png"
            ),

        cl.Starter(
            label="Što i kako da pitam - pomoć u korištenju!",
            message="Napiši pet pitanja koja ti mogu postaviti i objsni odgovore kakve daješ na njih",
            icon="/public/help.png"
            ),
        cl.Starter(
            label="Potencijali kojima ističe ugovor!",
            message="Koja 3 korisnika (oib-a) imaju najviše pretplatnika kojima ističe ugovor (fees_to_charge) za manje od 3 mjeseca.",
            icon="/public/business.png"
            )
        ]




@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():



    cl.user_session.set("chain", agent2, )





@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    import openai

    try:
        # Vaš kod koji može izazvati BadRequestError
        resp1 = await chain.achat(message.content)
        resp=resp1.response


    except openai.BadRequestError as e:
       
       resp="Pojavila se greška {e} na endpoint strani, ali ne brini, samo napiši novi upit!"

    

    msg = cl.Message(content=resp)

    #print(msg.content)
    await msg.send()

# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import chainlit as cl  # importing chainlit for our app
from dotenv import load_dotenv

from utils.nnn_tools import agent2
import os
import json

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



# File path for the JSON file
json_file_path = 'feedback_data.json'

# Function to read existing data from JSON file
def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return []

# Function to write data to JSON file
def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Define the feedback button callback
@cl.action_callback("feedback_button")
async def on_feedback(action):
    # Retrieve the last question and answer from user session
    last_interaction = cl.user_session.get("last_interaction")
    if last_interaction:
        # Add feedback to the interaction
        last_interaction['feedback'] = action.value
        # Read existing data
        data = read_json_file(json_file_path)
        # Append new interaction
        data.append(last_interaction)
        # Write updated data back to the file
        write_json_file(json_file_path, data)
        await cl.Message(content="Thank you for your feedback!").send()

        # Remove feedback buttons
        feedback_buttons = cl.user_session.get("feedback_buttons")
        if feedback_buttons:
            for button in feedback_buttons:
                await button.remove()
    else:
        await cl.Message(content="No interaction found to add feedback to.").send()



@cl.on_chat_start
async def start():

    cl.user_session.set("chain", agent2)





@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    resp1 = await chain.achat(message.content)

    resp=resp1.response

    actions = [
        cl.Action(name="feedback_button", value="positive", label="👍", description="Thumbs up"),
        cl.Action(name="feedback_button", value="negative", label="👎", description="Thumbs down")
    ]


    msg = cl.Message(content=resp, actions=actions)

    #print(msg.content)
    await msg.send()

    # Store the question and answer in user session for later feedback
    cl.user_session.set("feedback_buttons", msg.actions)
    cl.user_session.set("last_interaction", {
        "question": message.content,
        "answer": resp,
        "used_tool": [r.tool_name for r in resp1.sources] if len(resp1.sources)>0  else ["n/a"],
        "raw_input": [r.raw_input for r in resp1.sources] if len(resp1.sources)>0  else ["n/a"],
        "context": [r.text for r in resp1.source_nodes] if len(resp1.source_nodes)>0  else ["n/a"],
        "context_score" : [r.score for r in resp1.source_nodes]  if len(resp1.source_nodes)>0  else 0
    })

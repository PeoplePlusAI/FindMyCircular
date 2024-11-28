import chainlit as cl
import selfRAGAgentHF as sra


def load_agent():
    agent = sra.selfRAGAgent()
    return agent


@cl.on_chat_start
async def on_chat_start():
    agent = load_agent()
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    inputs = {
        "question": message.content,
        "iterations": 0,
    }
    async for output in agent.astream(inputs):
        for key, value in output.items():
            print("---" * 5)
            print(f"{key}: {value}")
            print("---" * 5)
            if key == "answer":
                await cl.Message(content=value["generation"]).send()

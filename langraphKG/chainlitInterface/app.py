import os

import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from load_dotenv import load_dotenv

load_dotenv()
ollama_base_url = ""

if os.getenv("BASE_URL"):
    ollama_base_url = os.getenv("BASE_URL")
else:
    ollama_base_url = None


@cl.step(type="VectorStore Initialization")
def load_vectorStore():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "yourpassword"

    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )

    vector_db = Chroma(
        collection_name="rag-chroma",
        persist_directory="./chroma",
        embedding_function=embeddings,
    )

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

    vecFromGraphDB = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="vecFromGraph",
        node_label="GovernmentDocument",
        text_node_properties=["summary"],
        embedding_node_property="embedding",
    )

    return vector_db, graph, vecFromGraphDB


@cl.step(type="Context Finder")
def find_context(user_query, vector_db, graph, vecFromGraphDB):
    simResults = vecFromGraphDB.similarity_search_with_relevance_scores(user_query)

    neo4j_query = f"MATCH (start {{name: '{simResults[0][0].metadata['name']}'}})<-[r]-(connected) RETURN connected"
    query_results = graph.query(neo4j_query)

    ret_content = vector_db.similarity_search(
        user_query, filter={"name": query_results[0]["connected"]["name"]}
    )

    return ret_content[0].page_content


@cl.on_chat_start
async def on_chat_start():
    vector_db, graph, vecFromGraphDB = load_vectorStore()
    cl.user_session.set("vector_db", vector_db)
    cl.user_session.set("graph", graph)
    cl.user_session.set("vecFromGraphDB", vecFromGraphDB)


@cl.on_message
async def main(message: cl.Message):
    vector_db = cl.user_session.get("vector_db")
    graph = cl.user_session.get("graph")
    vecFromGraphDB = cl.user_session.get("vecFromGraphDB")
    user_query = message.content
    context = find_context(user_query, vector_db, graph, vecFromGraphDB)

    llm = ChatOllama(
        base_url=ollama_base_url,
        model="llama3.1",
        temperature=0,
        num_ctx=16000,
        verbose=True,
    )

    prompt = PromptTemplate(
        template="""You are an legal question-answering agent. Use the following pieces of retrieved context which are part of circulars by the government and your experience to understand to create an answer relevant to the query containing all pertinent information required to answer the question. Make sure the answer is of an appropriate length containing specifics about the query's response. If you don't know the answer, just say that you don't know. Please only provide the answer in the output.

        Question: {question} 

        Context: {context} 

        Answer:""",
        input_variables=["question", "context"],
    )

    rag = prompt | llm | StrOutputParser()
    output = rag.invoke({"question": user_query, "context": context})
    await cl.Message(content=output).send()

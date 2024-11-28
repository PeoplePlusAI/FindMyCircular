import os
from typing import List

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()
ollama_base_url = ""

if os.getenv("BASE_URL"):
    ollama_base_url = os.getenv("BASE_URL")
else:
    ollama_base_url = None

retrieval_grader_llm = "llama3.1"
rag_chain_llm = "llama3.1"
hallucination_grader_llm = "llama3.1"
answer_grader_llm = "llama3.1"
question_rewriter_llm = "llama3.1"


def retrieval_grader_chain():
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        Use logic and understand the context of the question and document to make decisions \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=retrieval_grader_llm,
        format="json",
        temperature=0,
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    return retrieval_grader


def rag_chain():
    prompt = PromptTemplate(
        template="""You are an legal question-answering agent. Use the following pieces of retrieved context which are part of circulars by the goverment and your experience to understand to create an answer relevant to the query containing all pertinant information required to answer the question. Make sure the answer is of an appropriate length containing specifics about the query's repsonse. If you don't know the answer, just say that you don't know. Please only provide the answer in the output.

        Question: {question} 

        Context: {context} 

        Answer:""",
        input_variables=["question", "context"],
    )

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=rag_chain_llm,
        # format="json",
        temperature=0,
    )

    rag = prompt | llm | StrOutputParser()

    return rag


def hallucination_grader_chain():
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=hallucination_grader_llm,
        format="json",
        temperature=0,
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    return hallucination_grader


def answer_grader_chain():
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=answer_grader_llm,
        format="json",
        temperature=0,
    )

    answer_grader = prompt | llm | JsonOutputParser()

    return answer_grader


def question_rewriter_chain():
    prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n 
        Only return the improved question as the output.""",
        input_variables=["generation", "question"],
    )

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=question_rewriter_llm,
        temperature=0,
    )

    question_rewriter = prompt | llm | StrOutputParser()

    return question_rewriter


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        iterations: number of iterations
    """

    question: str
    generation: str
    documents: List[str]
    iterations: int


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    docsearch = Chroma(
        collection_name="rag-chroma",
        persist_directory="./chroma",
        embedding_function=embeddings,
    )

    retriever = docsearch.as_retriever()

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]
    state["iterations"] += 1

    rag = rag_chain()
    generation = rag.invoke({"question": question, "context": documents})
    print(generation)

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "iterations": state["iterations"],
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    retrieval_grader = retrieval_grader_chain()

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")

    question = state["question"]
    documents = state["documents"]

    question_rewriter = question_rewriter_chain()
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_grader = hallucination_grader_chain()
    answer_grader = answer_grader_chain()

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        if state["iterations"] >= 5:
            return "stop"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            print(f"---ITERATION COUNT: {state['iterations']}---")
            return "not supported"


def answer(state):
    """
    Return the answer

    Args:
        state (dict): The current graph state

    Returns:
        str: The answer

    TODO: Check the prev state to augment the answer
    """

    print("---ANSWER---")
    return state


def selfRAGAgent():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("answer", answer)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": "answer",
            "not useful": "transform_query",
            "stop": "answer",
        },
    )
    workflow.add_edge("answer", END)

    app = workflow.compile()
    return app

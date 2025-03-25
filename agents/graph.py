from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pprint import pprint
import time
from typing import List, Sequence
from typing_extensions import TypedDict, Annotated
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph, END
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# local imports 
from chains.retriever_qn import formulate_retriever_qn
from chains.response import rag_chain

from dotenv import load_dotenv
load_dotenv()
from llm_init import llm

# initialize vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = PineconeVectorStore(
    embedding=embeddings,
    index_name = "pricing-engine"
)
retriever = store.as_retriever()

# -------------------------------------------GRAPH----------------------------------------------------

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        messages: chat history
        summary: conversation summary
    """

    question: str
    generation: str
    summary: str
    documents: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]


def formulate_qn(state: GraphState):
    """history aware retrieval"""

    question = state["question"]
    messages = state["messages"]

    new_question = formulate_retriever_qn.invoke(
        {"messages": messages, "question": question}
    )

    return {"question": new_question}

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

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """
    Generate answer from the vector store 

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]

    # RAG generation
    generation = rag_chain.invoke(
        {
            "messages": messages, 
            "context": documents, 
            "question": question
        }
    )
    return {
        "documents": documents, 
        "question": question, 
        "generation": generation,
        "messages": [HumanMessage(content=question), AIMessage(content=generation)]
    }

def summarize_conversation(state: GraphState):

    messages = state["messages"]
    summary = state.get("summary", "")

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        if summary:
            # If a summary already exists, we use a different system prompt
            # to summarize it than if one didn't
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        print("---SUMMARIZING CONVERSATION---")
        response = llm.invoke(messages)
       
        # preserve the last 2 messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    

# -------------------------COMPILE THE GRAPH--------------------------------------------
workflow = StateGraph(GraphState)

# nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("history_aware_retrieval", formulate_qn)
workflow.add_node("summarize", summarize_conversation)

# Build and Compile
workflow.add_edge(
    START,
    "summarize"
)
workflow.add_edge(
    "summarize",
    "history_aware_retrieval"
)
workflow.add_edge(
    "history_aware_retrieval",
    "retrieve"
)
workflow.add_edge(
    "retrieve",
    "generate"
)
workflow.add_edge(
    "generate",
    END
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc345"}}
    question = "What is the withdrawal fee charged by CBZ for an individual account?"
    inputs = {"messages": [], "question": question}
    for output in app.stream(inputs, config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    res = value["generation"]
    print(res)

    # print(app.get_graph().draw_mermaid())
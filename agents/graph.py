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
from langgraph.prebuilt import ToolNode, tools_condition

# local imports 
from agents.chains.retriever_qn import formulate_retriever_qn
from agents.chains.response import rag_chain, multi_step_rag_chain, tools, llm_with_tools
from agents.chains.rewriter import query_rewriter
from agents.chains.router import query_router

from dotenv import load_dotenv
load_dotenv()
from agents.llm_init import llm

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

def router(state: GraphState):
    """query router"""

    question = state["question"]
    messages = state["messages"]

    print("---ROUTE QUESTION---")
    res = query_router.invoke(
        {
            "messages": messages,
            "question": question
        }
    )

    if res.datasource == "basic_retrieval":
        print("---ROUTE QUESTION TO BASIC---")
        return "basic"
    if res.datasource == "multi_step_retrieval":
        print("---ROUTE QUESTION TO MULTI STEP---")
        return "multi_step"
    
def rewriter(state: GraphState):
    """rewrites prompts for the multistep retrieval process"""

    question = state["question"]
    messages = state["messages"]

    print("-----INITIATING MULTI-STEP RETRIEVAL-----")
    print("---REWRITING THE QUERY---")
    res = query_rewriter.invoke(
        {
            "messages": messages,
            "question": question
        }
    )

    return{ "question": res.content }


def formulate_qn(state: GraphState):
    """history aware retrieval"""

    print("-----INITIATING BASIC RETRIEVAL-----")
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
    print("---GENERATE(BASIC RETRIEVAL)---")
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

def generate_2(state: GraphState):
    """
    Generate answer from the vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE(MULTI-STEP RETRIEVAL)---")
    question = state["question"]
    messages = state["messages"]

    # RAG generation
    # generation = multi_step_rag_chain.invoke(
    #     {
    #         "messages": messages,
    #         "question": question
    #     }
    # )
    generation = multi_step_rag_chain.invoke(
        {
            "messages": messages,
            "question": question
        }
    )
    return {
        "question": question, 
        "generation": generation.content,
        "messages": [HumanMessage(content=question), generation]
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
    
# def should_continue(state: GraphState):
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     return END
    

# -------------------------COMPILE THE GRAPH--------------------------------------------
workflow = StateGraph(GraphState)
tool_node = ToolNode(tools)

# nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("history_aware_retrieval", formulate_qn)
workflow.add_node("summarize", summarize_conversation)
workflow.add_node("rewriter", rewriter)
workflow.add_node("response", generate_2)
workflow.add_node("tools", tool_node)

# Build and Compile
workflow.add_edge(
    START,
    "summarize"
)
workflow.add_conditional_edges(
    "summarize",
    router,
    {
        "basic" : "history_aware_retrieval",
        "multi_step" : "rewriter"
    }
)
workflow.add_edge(
    "history_aware_retrieval",
    "retrieve"
)
workflow.add_edge(
    "rewriter",
    "response"
)
workflow.add_conditional_edges(
    "response", 
    tools_condition
)
workflow.add_edge(
    "tools",
    "response"
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
    question = "calculate the average USD amount charged by POSB and CABS for a Mini Statement using mobile banking"
    inputs = {"messages": [], "question": question}
    # for output in app.stream(inputs, config):
    #     for key, value in output.items():
    #         # Node
    #         pprint(f"Node '{key}':")
    #         # Optional: print full state at each node
    #         pprint(value["keys"], indent=2, width=80, depth=None)
    #     pprint("\n---\n")

    # # Final generation
    # res = value["generation"]
    # print(res)

    # print(app.get_graph().draw_mermaid())

    for chunk in app.stream(
    inputs, config, stream_mode="values"
    ):
        # chunk["messages"][-1].pretty_print()
        print(chunk['messages'])
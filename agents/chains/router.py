# chooses the action to take based on user input
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from llm_init import llm
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant retrieval process."""

    datasource: Literal["basic_retrieval", "multi_step_retrieval"] = Field(
        description="Given a user question choose to route it to basic_retrieval or multi_step_retrieval",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

# prompts
system = """
    You are a Query Routing Agent responsible for analyzing user queries and determining the best processing strategy. Your goal is to route each query to one of the following paths:

    1. **basic_retrieval:**  
    - Use this path for straightforward queries that require a direct, factual response—such as retrieving data from a local vector store containing previously ingested or domain-specific content.  
    - Examples include requests for specific charges, simple summaries, or historical data (e.g., "What is the fee for internal transfers at Bank X?" or "Summarize the bill payment charges for Bank Y").

    2. **multi_step_retrieval:**  
    - Use this path when the query is complex and involves multiple steps, such as comparing details between two banks or products, or when a nuanced analysis is needed.   
    - Example: "Compare the internal transfer charges between Bank A and Bank B."

    **Key Behaviors:**

    - **Select a Single Action:** Never attempt to perform both basic and multi-step retrieval simultaneously. Always choose the most appropriate action based on the query’s complexity.

    By following these guidelines, you will ensure that each query is processed using the most effective method.
    """

# Question Router
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "User query: {question}"),
    ]
)

question_router = route_prompt | structured_llm_router

if __name__ == "__main__":
    question = "How much does CBZ charge individuals for a Telegraphic Transfer"
    res = question_router.invoke(
        {
            "messages": [
            ],
            "question": question,
        }
    )
    print(res.datasource)

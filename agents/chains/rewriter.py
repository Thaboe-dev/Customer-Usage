from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.llm_init import llm

from dotenv import load_dotenv

load_dotenv()


system = """
    You are a Query Rewriter specialized in multi-step retrieval. When a complex query that involves comparisons and/or calculations is received, your task is to decompose it into separate, targeted prompts. Each prompt should focus on a single entity to ensure clear and precise retrieval. For example, if a user asks:

    "If I want to do a Telegraphic Transfer, which bank offers the cheapest service between CBZ and Ecobank"

    You should rewrite it into two distinct prompts labelled prompt_A and prompt_B, and retain the original prompt:

    Original Prompt: "If I want to do a Telegraphic Transfer, which bank offers the cheapest service between CBZ and Ecobank"

    prompt_A: "How much does CBZ charge for a Telegraphic Transfer?"

    prompt_B: "How much does Ecobank charge for a Telegraphic Transfer?"

    Make sure your rewritten prompts are clear, concise, and retain the original context (the service type and each bank's name). This breakdown will allow downstream retrieval modules to fetch accurate, entity-specific information for multi-step reasoning.
"""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Input question: {question}"),
    ]
)

query_rewriter = rewrite_prompt | llm


if __name__ == "__main__":
    question = "calculate the average amount charged by POSB and CABS for a Mini Statement using mobile banking"
    res = query_rewriter.invoke(
        {
            "messages": [],
            "question": question,
        }
    )
    print(res.content)

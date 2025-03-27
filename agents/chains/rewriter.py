from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from llm_init import llm

from dotenv import load_dotenv

load_dotenv()


system = """
    You are a Query Rewriter specialized in multi-step retrieval. When a complex query that involves comparisons is received, your task is to decompose it into separate, targeted prompts. Each prompt should focus on a single entity to ensure clear and precise retrieval. For example, if a user asks:

    "If I want to do a Telegraphic Transfer, which bank offers the cheapest service between CBZ and Ecobank"

    You should rewrite it into two distinct prompts:

    "How much does CBZ charge for a Telegraphic Transfer?"

    "How much does Ecobank charge for a Telegraphic Transfer?"

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
    question = "I want to do a RTGS transfer, using USD but Im not sure which bank to use between POSB and CABS. Please assist me"
    res = query_rewriter.invoke(
        {
            "messages": [],
            "question": question,
        }
    )
    print(res.content)

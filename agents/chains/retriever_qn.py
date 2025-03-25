from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from llm_init import llm
# from llm_init import llm

from dotenv import load_dotenv

load_dotenv()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("messages"),
        ("human", "{question}"),
    ]
)

formulate_retriever_qn = contextualize_q_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    question = ""
    res = formulate_retriever_qn.invoke(
        {
            "messages": [],
            "question": "Which bank provides the lowest withdrawal fees for individual accounts",
        }
    )
    print(res)

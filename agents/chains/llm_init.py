from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1
)

# llm = ChatGroq(
#     model_name = "deepseek-r1-distill-qwen-32b",
#     temperature=0
# )
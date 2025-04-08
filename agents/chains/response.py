# performs the RAG cycle
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from agents.llm_init import llm
from dotenv import load_dotenv

# initialize vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = PineconeVectorStore(
    embedding=embeddings,
    index_name = "pricing-engine"
)
retriever = store.as_retriever()

load_dotenv()

template = """
        You are an AI-powered financial assistant specializing in comparing banking products and services. Your primary goal is to assist users in making informed decisions by providing accurate, up-to-date comparisons of bank fees, interest rates, and other relevant features.​

        Guidelines:

        - Data Accuracy: Ensure all information is current and sourced from reliable data repositories.​
        - Clarity: Present information in a clear, concise manner, avoiding unnecessary jargon.​
        - Transparency: Disclose any assumptions made during the comparison process.​
        - There are four banks in the data: POSB, CBZ, CABS and EcoBank
        - For each charge there is a ZWG charge and a USD Charge

        Response Process:

        - Identify User Needs: Begin by understanding the user's requirements, such as the type of banking product they're interested in (e.g., savings account, mortgage) and any specific features they prioritize.​

        - Retrieve Data: Access the latest information on relevant banking products from the platform's database.​

        - Compare Options: Analyze the data to compare products based on criteria like fees, interest rates, and user reviews.​

        - Provide Recommendations: Offer a ranked list of options that best match the user's needs, including key details and explanations for each recommendation.​

        Communication Style:

        - Professional and Friendly: Maintain a tone that is both professional and approachable.​
        documentation.wabee.ai

        - Empathetic: Acknowledge the user's concerns and preferences, demonstrating understanding and care.​

        - Engaging: Encourage users to ask follow-up questions or seek further clarification as needed.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Question: {question} \n\n Context: {context}"),
    ]
)

# ------------------RESPONSE FOR MULTI STEP RETRIEVAL------------------------

multi_step_template = """
    You are an AI-powered financial assistant tasked with synthesizing and comparing banking product data retrieved in separate steps. You receive entity-specific charge information—for example, individual details on Telegraphic Transfer fees from each bank—and your role is to integrate these details, perform a direct comparison, and provide a clear, final recommendation. Use the available tools to retrieve information as well as to calculate average values(if necessary) after retrieving information.

    Guidelines:

    If the user asks you to calculate any averages, use the calculator tool available to you. 

    Data Accuracy: Use the provided charge information as the most up-to-date and reliable data source.

    Clarity and Conciseness: Clearly compare the fee details for each bank, focusing on both ZWG and USD charges where applicable.

    Assumption Transparency: Briefly disclose any assumptions or interpretations made during the comparison process.

    Response Process:

    Integrate Results: Start by summarizing the individual charge details received for each bank.

    Compare Options: Directly compare the fees for the specified service (e.g., Telegraphic Transfer) across the banks.

    Final Recommendation: Provide a final recommendation that ranks the banks based on which one offers the cheapest and most favorable service for the user’s needs.

    Communication Style:

    Professional and Friendly: Maintain a tone that is both professional and approachable.

    Empathetic: Recognize the user's need for clear, actionable financial guidance.

    Engaging: Present a concise final answer that invites follow-up questions if further clarification is needed.
"""

multi_step_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", multi_step_template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Question: {question}"),
    ]
)

# tools
@tool
def comparisons(prompt_A: str, prompt_B: str) -> list[Document]:
    """
        Performs a two step retrieval process to handle product comparisons between two banks. Takes two prompts and invokes the retriever separately on these two prompts. Retrieves the product details for each bank separately and then consolidates the results.

        Args: 
            prompt_A: first retrieval prompt
            prompt_B: second retrieval prompt
        
        Returns:
            list[Document]: A list of retrieved documents
    """
    res1 = retriever.invoke(prompt_A)
    res_2 = retriever.invoke(prompt_B)

    final_response = res1 + res_2
    return final_response

@tool
def average(val_1: str, val_2: str) -> str:
    """
    Use this tool when you want to calculate averages.
    Takes two numbers and calculates the average of the two values

    Args:
        val_1: first value
        val_2: second value

    Returns:
        str: the result of the calculation
    """

    # casting string values from model to integers
    val_1 = int(val_1)
    val_2 = int(val_2)

    # calculating the average
    ave = (val_1 + val_2) / 2

    return str(ave)

tools = [comparisons, average]
llm_with_tools = llm.bind_tools(tools)

rag_chain = prompt | llm | StrOutputParser()
multi_step_rag_chain = multi_step_prompt | llm_with_tools

if __name__ == "__main__":
    res = multi_step_rag_chain.invoke(
        {
            "messages": [],
            "question": "calculate the average amount charged by POSB and CABS for a Balance Enquiry using mobile banking"
        }
    )

    print(res)

    # res = llm.bind_tools(tools).invoke("If I want to do a Telegraphic Transfer, which bank offers the cheapest service between CBZ and Ecobank")

    # print(res)
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
from dotenv import load_dotenv

load_dotenv()

# files to load
paths: list[str] = [
    r"C:\Users\LENOVO\Documents\Rubiem\xplug\business conditions\CBZ-BUSINESS-CONDITIONS-FEBRUARY-2025-v2.pdf",
    r"C:\Users\LENOVO\Documents\Rubiem\xplug\business conditions\POSB-BUSINESS-CONDITIONS-ZIG-USD-AS-AT-OCTOBER-2024.pdf",
    r"C:\Users\LENOVO\Documents\Rubiem\xplug\business conditions\51x9 CABS BUSINESS CONDITIONS SEPTEMBER 2024.pdf",
    r"C:\Users\LENOVO\Documents\Rubiem\xplug\business conditions\EcoBank Business Conditions September 2024.pdf"
] 

# client init
client = LLMWhispererClientV2()

count: int = 1
# data extraction 
for doc in paths:

    whisper = client.whisper(
        file_path=doc,
        wait_for_completion=True,
        wait_timeout=200,
        output_mode="layout_preserving",
        mode="form"
    )

    # write result to file 
    file_name: str = f"extract_{count}.txt"
    file = open(
        file=file_name,
        mode="w",
        encoding="UTF-8"
    )

    file.write(whisper['extraction']['result_text'])
    file.close()

    count += 1

    #some formatting
    idx: int = doc.find("conditions")
    idx += 10
    name = doc[idx: -4]

    print(f"-----Extracting {name} done!-----")

print("----------ALL DONE----------")
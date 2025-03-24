import tiktoken

encoder = tiktoken.encoding_for_model("text-embedding-3-small")

docs: list[str] = [
    "extract_4.txt",
    
]

text: str = ""
for doc in docs:
    with open(doc) as f:
        txt = f.read()
    text += txt


tokens = encoder.encode(text)  # Exact token count
print(len(tokens))
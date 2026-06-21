from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3:mini")
response = llm.invoke("What is market research?")
print(response)
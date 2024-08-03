import chromadb
import replicate

# Get query from user
query = input("What is your query:\n")


# Initialize collection for db requests
client = chromadb.PersistentClient(
    path="/Users/anavbo/Desktop/Personal/skin-cancer-detection-app/RAG/chromadb"
)
collection = client.get_collection("skin-cancer")

# Query the collection and form the context
context = ""
retrieval = collection.query(query_texts="actinic kerosis", n_results=5)
for i in range(len(retrieval["documents"][0])):
    context += f"Document {i}: {retrieval["documents"][0][i]}\n"

# Initialize the LLM
api = replicate.Client(api_token="r8_XJns3nBIXao9KgeqbdqSvgCTP8JDf1u18BiTx")

# Prompt the LLM
system_prompt = "The following is contaxt that should be used to answer the prompt. the context starts from most relevant to least relevant." + context
output = api.run(
    "meta/llama-2-70b-chat",
    input={
        "top_k": 0,
        "top_p": 1,
        "prompt": query,
        "max_tokens": 512,
        "temperature": 0.5,
        "system_prompt": system_prompt,
        "length_penalty": 1,
        "max_new_tokens": 500,
        "min_new_tokens": -1,
        "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
        "presence_penalty": 0,
        "log_performance_metrics": False
    }
)
print("".join(output))

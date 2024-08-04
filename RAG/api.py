from flask import Flask, request
import replicate
import chromadb

app = Flask(__name__)

app.cache = {}
app.cache["client"] = chromadb.PersistentClient(
    path="/Users/anavbo/Desktop/Personal/skin-cancer-detection-app/RAG/chromadb"
)
app.cache["collection"] = app.cache["client"].get_collection("skin-cancer")
print("done")

# Initialize the LLM
app.cache["api"] = replicate.Client(api_token="r8_K5dHtj5QxO1EIyJjryMvOps4n8nwHWi44mQqT")


@app.route("/")
def rag_pipeline():
    print("request recieved")
    query = request.args["query"]
    # Query the collection and form the context
    
    context = ""
    retrieval = app.cache["collection"].query(query_texts="actinic kerosis", n_results=3)
    for i in range(len(retrieval["documents"][0])):
        context += f"Document {i}: {retrieval["documents"][0][i]}\n"

    # Prompt the LLM
    # try:
    system_prompt = "The resposne should be ONLY ONE TO TWO SENTENCES. Only use sentences and no styling. Use the following context to craft the best answer: " + context
    try:
        output = app.cache["api"].run(
            "meta/llama-2-70b-chat",
            input={
                "top_k": 0,
                "top_p": 1,
                "prompt": query,
                "max_tokens": 50,
                "temperature": 0.5,
                "system_prompt": system_prompt,
                "length_penalty": 1,
                "min_new_tokens": -1,
                "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                "presence_penalty": 0,
                "log_performance_metrics": False
            }
        )
        output_str = "".join(output)
    except Exception as e:
        output_str = f"Output Generation has reached $1,000,000. No more output can be generated. Please try again later."
    print("THE QUERY IS")
    print(query)
    print(output_str)
    return {"message": output_str}, 200

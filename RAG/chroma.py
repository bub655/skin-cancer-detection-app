import chromadb

query = input("What is your query:\n")


client = chromadb.PersistentClient(
    path="/Users/anavbo/Desktop/Personal/skin-cancer-detection-app/RAG/chromadb"
)
collection = client.get_collection("skin-cancer")

print(collection.query(query_texts=query, n_results=5)["data"])

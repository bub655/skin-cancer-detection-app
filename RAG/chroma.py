import chromadb

client = chromadb.PersistentClient(
    path="/Users/anavbo/Desktop/Personal/skin-cancer-detection-app/RAG/chromadb"
)
collection = client.get_collection("my_collection")

print(collection.query(query_texts="yello", n_results=3))

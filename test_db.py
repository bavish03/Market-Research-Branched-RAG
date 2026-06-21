import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
for collection_name in ["companies", "industry", "news"]:
    collection = client.get_collection(collection_name)
    print(collection_name, collection.count())
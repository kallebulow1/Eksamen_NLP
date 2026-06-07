from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import shutil

# ================================
# 1. INDLÆS DATA
# ================================
print("Indlæser data...")
df = pd.read_csv(
    r"C:\Users\mehdi\Documents\Dataanalyse 2 semester\Valgfag NLP\Uge 21\Torsdag\elg_power_metadata.csv",
    encoding="utf-8-sig"
)
print(f"Antal reviews: {len(df)}")
print(df['brand'].value_counts())

# ================================
# 2. EMBEDDING MODEL
# ================================
print("\nLoader embedding model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ================================
# 3. SLET GAMMEL DATABASE
# ================================
db_location = "./elg_power_chroma_db"
if os.path.exists(db_location):
    shutil.rmtree(db_location, ignore_errors=True)
    print("Gammel database slettet")

# ================================
# 4. OPRET DOCUMENTS
# brand i metadata = vi kan filtrere på Elgiganten vs Power
# ================================
documents = []
ids = []

print("\nOpretter documents...")
for i, row in df.iterrows():
    content = f"""
    Brand: {row['brand']}
    Anmeldelse: {row['content']}
    Titel: {row['title']}
    Rating: {row['rating']}
    Sentiment: {round(float(row['sscore']), 2)}
    Køn: {row['gender']}
    År: {row['year']}
    """

    document = Document(
        page_content=content,
        metadata={
            "brand":     str(row["brand"]),
            "rating":    int(row["rating"]),
            "sentiment": round(float(row["sscore"]), 2),
            "gender":    str(row["gender"]),
            "year":      int(row["year"])
        },
        id=str(i)
    )
    documents.append(document)
    ids.append(str(i))

    if i % 500 == 0:
        print(f"  {i}/{len(df)} documents behandlet...")

print(f"\nAlt i alt: {len(documents)} documents")

# ================================
# 5. OPRET CHROMA DATABASE
# ================================
print("\nOpretter Chroma database...")
vector_store = Chroma(
    collection_name="elg_power_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

print("Tilføjer documents (tager tid)...")
vector_store.add_documents(documents=documents, ids=ids)
print("Database klar!")

# ================================
# 6. TEST
# ================================
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("dårlig levering")
print(f"\nTest: Fandt {len(docs)} relevante anmeldelser")
for doc in docs[:2]:
    print(doc.page_content[:200])
    print(f"Metadata: {doc.metadata}")
    print("---")
print("\nKlar til chatbot!")

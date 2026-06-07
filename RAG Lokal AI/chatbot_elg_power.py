from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ================================
# 1. LOAD DATABASE
# ================================
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="elg_power_reviews",
    persist_directory="./elg_power_chroma_db",
    embedding_function=embeddings
)

# ================================
# 2. SMART RETRIEVER
# Fix: hvis BEGGE brands nævnes → ingen brand-filter
# ================================
def get_retriever(question):
    q = question.lower()
    brand_filter  = None
    gender_filter = None

    # Brand filter
    begge_brands = "elgiganten" in q and "power" in q
    if not begge_brands:
        if "elgiganten" in q:
            brand_filter = {"brand": "Elgiganten"}
            print("  → Filtrerer på: Elgiganten")
        elif "power" in q:
            brand_filter = {"brand": "Power"}
            print("  → Filtrerer på: Power")

    # Køn filter
    if any(w in q for w in ["kvinde", "kvinder", "hun", "hende"]):
        gender_filter = {"gender": "F"}
        print("  → Filtrerer på: Kvinder")
    elif any(w in q for w in ["mænd", "mand", "han", "ham"]):
        gender_filter = {"gender": "M"}
        print("  → Filtrerer på: Mænd")

    # Kombiner filtre korrekt
    if brand_filter and gender_filter:
        # Chroma kræver $and når der er flere filtre
        final_filter = {"$and": [brand_filter, gender_filter]}
    elif brand_filter:
        final_filter = brand_filter
    elif gender_filter:
        final_filter = gender_filter
    else:
        final_filter = None
        print("  → Ingen filter — henter bredt")

    if final_filter:
        return vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": final_filter}
        )
    return vector_store.as_retriever(search_kwargs={"k": 5})

# ================================
# 3. MODEL OG PROMPT
# Kortere prompt = hurtigere svar
# ================================
model = OllamaLLM(model="llama3.2")

template = """

REGLER:
- Svar på dansk
- Brug KUN de anmeldelser du får
- Vær konkret og kortfattet
- Sammenlign brands når begge er til stede

ANMELDELSER:
{reviews}

SPØRGSMÅL: {question}

SVAR:

**Konklusion:** [1-2 sætninger]

**Ros:** [Hvad roser kunderne?]

**Kritik:** [Hvad klager kunderne over?]

**Elgiganten vs Power:** [Forskelle hvis begge brands er med]
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ================================
# 4. FEDT TERMINAL UI
# ================================
def print_header():
    print("\n" + "█"*72)
    print("█" + " "*70 + "█")
    print("█" + "   🛒  ELGIGANTEN VS POWER CHATBOT  🛒   ".center(70) + "█")
    print("█" + " "*70 + "█")
    print("█"*72)

def print_reviews(docs):
    print("\n┌─ FUNDNE ANMELDELSER " + "─"*30)
    for i, doc in enumerate(docs):
        # Udtræk brand og rating fra metadata
        brand  = doc.metadata.get('brand', '?')
        rating = doc.metadata.get('rating', '?')
        gender = doc.metadata.get('gender', '?')
        year   = doc.metadata.get('year', '?')

        # Kort uddrag af anmeldelsen
        tekst = doc.page_content
        # Find selve anmeldelsesteksten
        lines = [l.strip() for l in tekst.split('\n') if 'Anmeldelse:' in l]
        uddrag = lines[0].replace('Anmeldelse:', '').strip()[:80] if lines else '...'

        # Farve baseret på brand
        mærke = "🔵" if brand == "Elgiganten" else "🔴"

        print(f"│ {mærke} [{i+1}] {brand} | ⭐{rating} | 👤{gender} | 📅{year}")
        print(f"│     \"{uddrag}...\"")
        print("│")
    print("└" + "─"*51)

def print_svar(result):
    print("\n╔" + "═"*70 + "╗")
    print("║" + " SVAR ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    
    for linje in result.split('\n'):
        if linje.strip():
            # Bryd lange linjer op i mindre stykker
            while len(linje) > 68:
                print("║ " + linje[:68] + " ║")
                linje = "  " + linje[68:]  # indent fortsættelse
            print("║ " + linje.ljust(68) + " ║")
    
    print("╚" + "═"*70 + "╝")

# ================================
# 5. CHATBOT LOOP
# ================================
print_header()
print("\n  Eksempel spørgsmål:")
print("  → Er Elgiganten eller Power bedst på levering?")
print("  → Hvad klager folk over hos Power?")
print("  → Hvad synes kvinder om Elgiganten?")
print("  → Hvad er den største forskel på de to brands?")

while True:
    print("\n" + "─"*72) 
    question = input("  💬 Spørgsmål (q = afslut): ")

    if question.lower() == "q":
        print("\n  Afslutter. Hej hej! 👋")
        break

    print("\n  🔍 Søger i anmeldelser...")
    retriever = get_retriever(question)
    reviews_docs = retriever.invoke(question)

    print(f"  ✅ Fandt {len(reviews_docs)} relevante anmeldelser")

    # Vis de fundne anmeldelser
    print_reviews(reviews_docs)

    reviews = "\n\n---\n\n".join(
        [f"Anmeldelse {i+1}:\n{doc.page_content}"
         for i, doc in enumerate(reviews_docs)]
    )

    print("\n  🤖 Llama tænker...\n")

    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })

    print_svar(result)

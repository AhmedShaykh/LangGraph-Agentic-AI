from langchain_text_splitters import RecursiveCharacterTextSplitter;
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings;
from langchain_community.document_loaders import PyPDFLoader;
from langchain_core.prompts import ChatPromptTemplate;
from langgraph.graph import StateGraph, START, END;
from typing_extensions import TypedDict;
from langchain_chroma import Chroma;
from dotenv import load_dotenv;
import chromadb;
import os;

load_dotenv();

PDF_PATH = "agent/AI.pdf";

COLLECTION_NAME = "langgraph_vectorembedding";

CHUNK_SIZE = 100;

CHUNK_OVERLAP = 10;

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
);

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
);

def collection_has_data(client, name):

    try:

        collection = client.get_collection(name=name);

        return collection.count() > 0;

    except Exception:

        return False;

collection_exists = collection_has_data(chroma_client, COLLECTION_NAME);

if not collection_exists:
    
    print(f"Collection `{COLLECTION_NAME}` Not Found. Creating {COLLECTION_NAME}...\n");

    loader = PyPDFLoader(PDF_PATH);

    docs = loader.load();

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    );
    
    chunks = splitter.split_documents(docs);

    vectorstore = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=chunks,
        embedding=embedding_model,
        client=chroma_client
    );

    print(f"Ingestion Complete. {len(chunks)} Chunks Stored.\n");

else:

    print(f"Collection `{COLLECTION_NAME}` Already Exists.\n");

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        client=chroma_client
    );

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
);

llm = ChatMistralAI(model="mistral-small-2506");

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context,
                say: I could not find the answer in the document."""),
    ("human", """Context:
                {context}

                Question:
                {question}""")
]);

class RAGState(TypedDict):
    question: str;
    context: str;
    answer: str;

def retrieve_node(state: RAGState) -> RAGState:

    docs = retriever.invoke(state["question"]);

    context = "\n\n".join([doc.page_content for doc in docs]);
    
    return {"context": context};

def generate_node(state: RAGState) -> RAGState:

    final_prompt = prompt.invoke({
        "context": state["context"],
        "question": state["question"]
    });

    response = llm.invoke(final_prompt);

    return {"answer": response.content};

graph_builder = StateGraph(RAGState);

graph_builder.add_node("retrieve", retrieve_node);

graph_builder.add_node("generate", generate_node);

graph_builder.add_edge(START, "retrieve");

graph_builder.add_edge("retrieve", "generate");

graph_builder.add_edge("generate", END);

rag_graph = graph_builder.compile();

print("RAG System Is Ready");

print("Press 0 To Exit\n");

while True:

    query = input("You: ");

    if query == "0":

        break;

    result = rag_graph.invoke({"question": query});

    print(f"\nAI: {result["answer"]}\n");
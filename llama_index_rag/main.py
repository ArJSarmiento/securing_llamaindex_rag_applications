from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.llms.ollama import Ollama


async def prompt(input: str):
    documents = SimpleDirectoryReader(
        input_files=["./llama_index_rag/paul_essay.txt"]
    ).load_data()
    print(documents)

    embed_model = HuggingFaceEmbedding()

    # create client
    db = chromadb.PersistentClient(path="./llama_index_rag/chroma_db")
    chroma_collection = db.get_or_create_collection("paul_collection")

    # save embedding to disk
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create index
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    # load from disk
    db2 = chromadb.PersistentClient(path="./llama_index_rag/chroma_db")
    chroma_collection = db2.get_or_create_collection("paul_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # query
    llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

    # Query Data from the persisted index
    query_engine = index.as_query_engine(llm=llm, streaming=True)

    streaming_response = query_engine.query(input)
    for text in streaming_response.response_gen:
        yield (text)


if __name__ == "__main__":
    input = "What is the essay about?"
    prompt(input)

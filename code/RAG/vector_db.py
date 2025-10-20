import chromadb
import os
import sys

# Add the Log directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Logs')))

from Logging import Log

class RAG:

    def __init__(self):
        context_path = os.path.join("..", "context-pages")

        context_pages = [
            {
                "id": "1",
                "document_name": "undergraduate",
                "source": "lcsee.statler.wvu.edu/undergraduate.txt",
                "document_path": os.path.join(context_path,"lcsee.statler.wvu.edu", "undergraduate.txt")
            },
            {
                "id": "2",
                "document_name": "alumni-and-friends",
                "source": "lcsee.statler.wvu.edu/alumni-and-friends.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "alumni-and-friends.txt")
            },
            {
                "id": "3",
                "document_name": "graduate",
                "source": "lcsee.statler.wvu.edu/graduate.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "graduate.txt")
            },
            {
                "id": "4",
                "document_name": "research",
                "source": "lcsee.statler.wvu.edu/research.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "research.txt")
            },
            {
                'id': "5",
                "document_name": "student-life",
                "source": "lcsee.statler.wvu.edu/student-life.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "student-life.txt")
            },
            { 
                'id': "6",
                "document_name": "faculty-staff",
                "source": "lcsee.statler.wvu.edu/faculty-staff.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "faculty-staff.txt")
            },
            { 
                'id': "7",
                "document_name": "lcsee.statler.wvu.edu",
                "source": "lcsee.statler.wvu.edu.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu.txt")
            }
            
        ]

        chunked_context = self.chunk_data(context_pages)

        chroma_client = chromadb.PersistentClient(path="../RAG/context_db")

        self.collection = chroma_client.get_or_create_collection(name="lcsee")
    
        if self.collection.count() > 0:
            Log.log("SYSTEM", "Collection already populated.")
        else:

            for chunk in chunked_context:
            
                self.collection.add(
                    ids=[chunk["chunk_id"]],
                    documents=[chunk["content"]],
                    metadatas=[{"source": chunk["metadata"]["source"], "document_name": chunk["metadata"]["document_name"]}],
                )

        

    def answer_question(self, question: str):
        
        result = self.collection.query(
            query_texts=[question],
            n_results=3
        )
        return result


    @staticmethod
    def chunk_data(data, chunk_size=500, overlap=50):
        """
        Function to chunk data into smaller pieces for better processing.
        Args:
            data: list - A list of dictionaries, each representing a page/document.
            chunk_size: int - the size of each chunk
            overlap: int - the overlap between chunks
        Returns:
            chunks: list - list of chunked data
        """
        chunks = []
        for page in data:
            try:
                with open(page["document_path"], "r", encoding="utf-8") as file:
                    document_content = file.read()
            except FileNotFoundError:
                # Handle cases where the file might be missing
                print(f"Warning: File not found at {page['document_path']}. Skipping this file.")
                continue  # Move to the next item in the loop

            start = 0
            chunk_index = 0
            while start < len(document_content):
                end = start + chunk_size
                chunk = document_content[start:end]
                chunks.append({
                    "id": page["id"],
                    "chunk_id": f"{page['id']}_chunk_{chunk_index}",
                    "content": chunk,
                    "metadata": {
                        "source": page["source"],
                        "document_name": page["document_name"]
                    }
                })
                chunk_index += 1
                start += chunk_size - overlap
                
        # This return statement is now correctly placed outside the loop.
        return chunks




def main():

    rag = RAG()

    print("Enter your question:")
    question = input()
    answer = rag.answer_question(question)

    print(f"Question: {question}")
    
    # Pretty-print the query result
    if isinstance(answer, dict):
        # chromadb returns lists-per-query, so take the first (and only) query result
        docs = answer.get("documents", [[]])[0]
        metas = answer.get("metadatas", [[]])[0]
        distances = answer.get("distances", [[]])[0]
        ids = answer.get("ids", [[]])[0]

        if not docs:
            print("No documents returned.")
        else:
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else None
                id_ = ids[i] if i < len(ids) else None
                print(f"\nResult #{i+1}")
                if id_ is not None:
                    print(f"  id: {id_}")
                if dist is not None:
                    print(f"  distance: {dist}")
                if meta:
                    src = meta.get("source")
                    name = meta.get("document_name")
                    print(f"  source: {src}  document_name: {name}")
                print("  content:")
                print(doc)
    else:
        # Fallback for unexpected result types
        print(answer)

if __name__ == "__main__":
    main()




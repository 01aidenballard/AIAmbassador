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
                "document_path": os.path.join(context_path,"lcsee.statler.wvu.edu", "alumni-and-friends.txt")
            },
            {
                "id": "2",
                "document_name": "alumni-and-friends",
                "source": "lcsee.statler.wvu.edu/alumni-and-friends.txt",
                "document_path": os.path.join(context_path, "lcsee.statler.wvu.edu", "undergraduate.txt")
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
            data: str - the data to be chunked
            chunk_size: int - the size of each chunk
            overlap: int - the overlap between chunks
        Returns:
            chunks: list - list of chunked data
        """
        chunks = []
        for page in data:
            with open(page["document_path"], "r", encoding="utf-8") as file:
                document_content = file.read()
            
            start = 0
            chunk_index = 1
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
            return chunks




def main():

    rag = RAG()

    print("Enter your question:")
    question = input()
    answer = rag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()




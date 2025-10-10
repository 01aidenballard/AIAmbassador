import chromadb
import os

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

        chroma_client = chromadb.Client()

        self.collection = chroma_client.create_collection(name="lcsee")

        for page in context_pages:

            with open(page["document_path"], "r", encoding="utf-8") as file:
                document_content = file.read()
            
            self.collection.add(
                ids=[page["id"]],
                documents=[document_content],
                metadatas=[{"source": page["source"], "document_name": page["document_name"]}],
            )

    def answer_question(self, question: str):
        
        result = self.collection.query(
            query_texts=[question],
            n_results=1
        )
        return result

def main():

    rag = RAG()

    print("Enter your question:")
    question = input()
    answer = rag.answer_question(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

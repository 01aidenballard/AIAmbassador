import chromadb
import os
import sys
import argparse
import json
import time
import psutil
import threading

from contextlib import contextmanager

# Add the Log directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Logs')))

from Logging import Log


#== Method ==#
def load_testset(path: str) -> dict:
    '''
    load the custom test dataset and label the questions for classification

    Args:
        path (str): path to custom test dataset

    Returns:
        dict: dataset with labeled questions
    '''
    # check if path exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    # load the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    data = data['data']

    # iterate through each label
    labeled_data = {'data': []}

    for ql in data: # iterate through each question/label
        # extract the label and q data
        dbanswer = ql['answer']
        question = ql['question']

        labeled_data['data'].append(
            {
                'question': question,
                'answer': dbanswer
            }
        )

    return labeled_data

class RAG:

    def __init__(self,):
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

        

    def answer_question(self, question: str, n_results: int = 2):
        
        result = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return result


    @staticmethod
    def chunk_data(data, chunk_size=500, overlap=50):
        """
        Function to chunk data into smaller pieces for better processing.
        Args:
            data: list - A list of dictionaries, each representing a page/document.
            chunk_size: int - the size of each chunk (unused in --- mode, but kept for compatibility)
            overlap: int - the overlap between chunks (unused here, but kept for compatibility)
        Returns:
            chunks: list - list of chunked data
        """
        chunks = []
        for page in data:
            try:
                with open(page["document_path"], "r", encoding="utf-8") as file:
                    document_content = file.read()
            except FileNotFoundError:
                print(f"Warning: File not found at {page['document_path']}. Skipping this file.")
                continue

            # Split on your section divider
            chunks_split = document_content.split('---')

            for chunk_index, chunk in enumerate(chunks_split):
                chunk = chunk.strip()

                # Only store non-empty chunks
                if chunk:
                    chunks.append({
                        "id": page["id"],
                        "chunk_id": f"{page['id']}_chunk_{chunk_index}",
                        "content": chunk,
                        "metadata": {
                            "source": page["source"],
                            "document_name": page["document_name"]
                        }
                    })

        return chunks
    
    def pretty_print(self, answer):
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

    def train(self):

        # Clear and repopulate the database
        Log.log("SYSTEM", "Clearing and repopulating the database.")

        self.collection.delete()

        chunked_context = self.chunk_data(self.context_pages)

        for chunk in chunked_context:

            self.collection.add(
                ids=[chunk["chunk_id"]],
                documents=[chunk["content"]],
                metadatas=[{"source": chunk["metadata"]["source"], "document_name": chunk["metadata"]["document_name"]}],
            )

        Log.log("SYSTEM", "Database repopulated successfully.")

@contextmanager
def cpu_usage_monitor(sample_interval=0.05):
    process = psutil.Process(os.getpid())
    memory_samples = []
    running = True

    # mem based sampling function
    def sample_memory():
        while running:
            memory_samples.append(process.memory_info().rss)
            time.sleep(sample_interval)

    # start memory sampling
    sampler_thread = threading.Thread(target=sample_memory)
    sampler_thread.start()

    # before execution
    start_wall = time.time()
    start_cpu = process.cpu_times().user + process.cpu_times().system
    start_mem = process.memory_info().rss
    process.cpu_percent(interval=None)

    try:
        # during execution
        yield_value = {}
        yield yield_value
    finally:
        # after execution
        end_wall = time.time()
        end_cpu = process.cpu_times().user + process.cpu_times().system
        end_mem = process.memory_info().rss
        end_cpu_percent = process.cpu_percent(interval=None)

        # stop memory sampling
        running = False
        sampler_thread.join()

        # calculate averages
        if memory_samples:
            avg_ram_usage_bytes = sum(memory_samples) / len(memory_samples)
        else:
            avg_ram_usage_bytes = 0

        # calc metrics
        wall_time_elapsed = end_wall - start_wall
        cpu_time_used = end_cpu - start_cpu
        cpu_utilization_percent = (cpu_time_used / wall_time_elapsed) * 100
        ram_used_bytes = end_mem - start_mem
        ram_used_mb = ram_used_bytes / (1024 * 1024)

        # store in dict
        yield_value.update({
            'wall_time': wall_time_elapsed,
            'cpu_time': cpu_time_used,
            'cpu_utilization_calculated': cpu_utilization_percent,
            'cpu_utilization_psutil': end_cpu_percent,
            'ram_usage_change_mb': ram_used_mb,
            'ram_usage_start_mb': start_mem / (1024 * 1024),
            'ram_usage_end_mb': end_mem / (1024 * 1024),
            'ram_usage_avg_mb': avg_ram_usage_bytes / (1024 * 1024)  # Convert to MB
        })

def main(args):

    rag = RAG()

    if args.test:
        print("Running on test dataset...")
        test_dataset = load_testset(os.path.join("..", "test_dataset.json"))

        count = 1

        total_distance = 0
        avg_time = 0
        avg_cpu_time = 0
        avg_cpu_usage = 0
        avg_ram_usage = 0

        for QA in test_dataset['data']:
            question = QA['question']
            dbanswer = QA['answer']
            
            
            with cpu_usage_monitor() as metrics:
                answer = rag.answer_question(question, n_results=1)
            
            avg_time += metrics['wall_time']
            avg_cpu_time += metrics['cpu_time']
            avg_cpu_usage += metrics['cpu_utilization_psutil']
            avg_ram_usage += metrics['ram_usage_avg_mb']

            print(f'Question: {question}\nAnswer: {answer}')
            print(f'Stats:')
            print(f'  Response time: {(metrics["wall_time"]):.2f} seconds')
            print(f'  CPU time taken: {(metrics["cpu_time"]):.2f} seconds')
            print(f'  CPU usage: {(metrics["cpu_utilization_psutil"]):.2f}%')
            print(f'  Avg RAM usage: {(metrics["ram_usage_avg_mb"]):.2f} MB\n')
            


            print(f"Question (#{count}): {question}\nDataset Answer: {dbanswer}\nTime Taken: {metrics['wall_time']}\nRAG Answer:")
            rag.pretty_print(answer)

        n = len(test_dataset['data'])
        avg_time /= n
        avg_cpu_time  /= n
        avg_cpu_usage /= n
        avg_ram_usage /= n
        avg_distnace = total_distance / n

        print('Overall Statistics:')
        print(f' Total questions answered: {n}')
        print(f' Average time taken for answering questions: {avg_time:.3f} seconds')
        print(f' Average CPU time taken for answering questions: {avg_cpu_time:.3f} seconds')
        print(f' Average CPU usage: {avg_cpu_usage:.2f}%')
        print(f' Average RAM usage: {avg_ram_usage:.2f} MB\n')
        print(f' Average distance of retrieved documents: {avg_distnace:.4f}')

    elif args.train:

        rag.train()
    
    else:

        print("Enter your question:")
        question = input()
        answer = rag.answer_question(question)
        print(f"Question: {question}")
        rag.pretty_print(answer)

    
    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='RAG Vector Database Interface')

    argparser.add_argument('--test', action='store_true', help='Run on test dataset')
    argparser.add_argument('--train', action='store_true', help='Clear and repopulate the database')

    args = argparser.parse_args()

    main(args)




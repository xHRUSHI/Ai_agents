import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model directly
model_name = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text generation pipeline (Adjust Parameters!)
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype="auto",
                device_map="auto",
                max_new_tokens=2048,    # Increased
                do_sample=True,
                top_p=0.95,         # Adjusted
                top_k=40,          # Adjusted
                temperature=0.2)    # Adjusted

llm = HuggingFacePipeline(pipeline=pipe)

class PDFQuestionAnsweringAgent:
    def __init__(self, file_path, file_type="pdf", chunk_size=750, chunk_overlap=150): # Adjusted Chunking
        self.file_path = file_path
        self.file_type = file_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db = self._create_knowledge_base()
        self.qa_chain = self._create_qa_chain()

    def _load_document(self): # (Same as before)
        if self.file_type == "pdf":
            loader = PyPDFLoader(self.file_path)
        elif self.file_type == "txt":
            loader = TextLoader(self.file_path)
        else:
            raise ValueError("Invalid file_type. Must be 'pdf' or 'txt'.")
        documents = loader.load()
        return documents

    def _split_text(self, documents): # (Same as before)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        return texts

    def _create_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",  # Keep trying different models
                                            model_kwargs={'device': 'cpu'})
        return embeddings

    def _create_knowledge_base(self):  # (Same as before)
        documents = self._load_document()
        texts = self._split_text(documents)
        embeddings = self._create_embeddings()
        db = FAISS.from_documents(texts, embeddings)
        return db

    def _create_qa_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Experiment with "map_reduce" or "refine"
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),  # Adjusted k value
            return_source_documents=False
        )
        return qa_chain

    def ask_question(self, question):  # (Same as before)
        try:
            answer = self.qa_chain.run(question)

            if answer and answer.strip():
                return answer
            else:
                return "I'm sorry, but the provided document does not contain information about this topic."
        except Exception as e:
            print(f"An error occurred: {e}")
            return "I encountered an error while trying to answer your question."

# Example Usage
if __name__ == "__main__":
    file_path = '/home/hrushi/workdir/AIagents/raw_data/product_data_old/HIDS_overview.txt'
    file_type = "txt"

    agent = PDFQuestionAnsweringAgent(file_path, file_type=file_type)

    # question1 = "What is the main purpose of the document?"
    question2 = "What is the benifits of using HIDS?"

    # answer1 = agent.ask_question(question1)
    answer2 = agent.ask_question(question2)

    # print(f"Question: {question1}")
    # print(f"Answer: {answer1}")

    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer2}")
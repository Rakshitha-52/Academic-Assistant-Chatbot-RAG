# ============================================================
# Domain-Specific Chatbot with RAG
# Domain: Academic Assistance
# ============================================================

import os
import fitz
import numpy as np
from dotenv import load_dotenv
import requests
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD PDF DOCUMENTS
# ============================================================

PDF_FOLDER = "pdfs/"

documents = []

for filename in os.listdir(PDF_FOLDER):

    if filename.endswith(".pdf"):

        filepath = os.path.join(PDF_FOLDER, filename)

        pdf = fitz.open(filepath)

        text = ""

        for page in pdf:
            text += page.get_text()

        documents.append({
            "topic": filename,
            "content": text
        })

print(f"\nLoaded {len(documents)} PDF files.")


# ============================================================
# 2. DOCUMENT RETRIEVER
# ============================================================

class DocumentRetriever:

    def __init__(self, knowledge_base, num_docs=2):

        self.knowledge_base = knowledge_base
        self.num_docs = num_docs

        self.texts = [doc['content'] for doc in knowledge_base]

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.texts
        )

        print("Documents Indexed Successfully")

    def retrieve(self, query):

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vec,
            self.tfidf_matrix
        ).flatten()

        top_indices = np.argsort(scores)[::-1][:self.num_docs]

        results = []

        for idx in top_indices:

            if scores[idx] > 0:

                results.append({

                    "topic": self.knowledge_base[idx]['topic'],

                    "content": self.knowledge_base[idx]['content'][:1500],

                    "score": round(float(scores[idx]), 4)
                })

        return results


# ============================================================
# 3. LLM GENERATOR
# ============================================================

from openai import OpenAI


class LLMGenerator:

    def __init__(self, api_key):

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def generate(self, query, context_docs):

        if not context_docs:
            return "No relevant academic information found."

        context = "\n\n".join(
            [
                f"{doc['topic']}:\n{doc['content']}"
                for doc in context_docs
            ]
        )

        prompt = f"""
You are an academic assistant chatbot.

Answer the student's question only from the provided context.

Context:
{context}

Question:
{query}

Give a simple and accurate academic answer.
"""

        response = self.client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content
# ============================================================
# 4. CHATBOT PIPELINE
# ============================================================

class AcademicChatbot:

    def __init__(self, api_key):

        self.retriever = DocumentRetriever(
            documents,
            num_docs=2
        )

        self.generator = LLMGenerator(api_key)

        print("\nAcademic Assistant Chatbot Ready")

    def chat(self, query):

        retrieved_docs = self.retriever.retrieve(query)

        print("\nRetrieved Documents:")

        for doc in retrieved_docs:

            print(
                f"{doc['topic']} | Score: {doc['score']}"
            )

        response = self.generator.generate(
            query,
            retrieved_docs
        )

        return response


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    bot = AcademicChatbot(API_KEY)

    print("\n===================================")
    print("Academic Assistant Chatbot")
    print("Type 'exit' to stop")
    print("===================================\n")

    while True:

        user_query = input("You: ")

        if user_query.lower() in ["exit", "quit", "bye"]:

            print("\nGoodbye!")
            break

        answer = bot.chat(user_query)

        print("\nBot:", answer)
        print("\n" + "-" * 50)


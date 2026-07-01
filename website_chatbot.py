"""
Website Chatbot Assessment - Single Python File Submission

Description:
This program scrapes the content of a website, processes the extracted text,
retrieves the most relevant website sections for a user query, and generates
an answer using an OpenRouter LLM. The chatbot runs entirely in the console.

Step-by-step process followed:
1. Set up Python environment and installed required libraries.
2. Used requests to fetch website HTML content.
3. Used BeautifulSoup to parse and extract useful text from the website.
4. Cleaned the extracted text by removing unnecessary whitespace.
5. Split the website text into smaller chunks for easier retrieval.
6. Implemented cosine similarity to find the most relevant chunks for a query.
7. Integrated OpenRouter API with the Qwen model to generate answers.
8. Restricted the chatbot to answer only from the provided website content.
9. Built a console-based chatbot loop for continuous interaction.

How to run:
1. Install required dependencies:
   pip install requests beautifulsoup4

2. (Optional) Configure settings in the CONFIGURATION section:
   - TARGET_URL: Set to the website you want to scrape (default: https://botpenguin.com/)
   - OPENROUTER_API_KEY: Update with your OpenRouter API key if needed

3. Run the chatbot:
   python website_chatbot.py

4. Interact with the chatbot:
   - Type your questions about the website content
   {
    What is this website about?
    What services does the company offer?
    What pricing plans are available?
    What features are mentioned?
    Who is the target audience?
    Is there contact information?
    }
   - Type 'exit' to quit the chatbot
"""

import os
import re
import math
import requests
from bs4 import BeautifulSoup
from collections import Counter


# -------------------------------------------------------

# CONFIGURATION
# -------------------------------------------------------

MODEL = "qwen/qwen3.6-plus:free"
TARGET_URL = "https://botpenguin.com/"   
OPENROUTER_API_KEY = "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"

REQUEST_TIMEOUT = 20
API_TIMEOUT = 30
CHUNK_SIZE = 800
TOP_K = 3


# -------------------------------------------------------
# SCRAPE WEBSITE CONTENT
# -------------------------------------------------------
def fetch_website_text(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe", "svg", "img", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = clean_text(text)

        if not text:
            raise ValueError("No usable text was extracted from the website.")

        return text

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch website content: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error while processing website content: {e}") from e


# -------------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------------
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


# -------------------------------------------------------
# CHUNK TEXT
# -------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# -------------------------------------------------------
# TOKENIZATION + VECTOR
# -------------------------------------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def text_to_vector(text):
    return Counter(tokenize(text))


# --------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------
def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[word] * vec2[word] for word in common)

    magnitude1 = math.sqrt(sum(value * value for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value * value for value in vec2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# -------------------------------------------------------------------
# RETRIEVE MOST RELEVANT CHUNKS
# -------------------------------------------------------------------
def retrieve_relevant_chunks(query, chunks, top_k=TOP_K):
    query_vector = text_to_vector(query)
    scored_chunks = []

    for chunk in chunks:
        chunk_vector = text_to_vector(chunk)
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    if not relevant_chunks:
        relevant_chunks = scored_chunks[:top_k]
        relevant_chunks = [chunk for _, chunk in relevant_chunks]

    return relevant_chunks


# -----------------------------------------------------------------------
# CALL OPENROUTER LLM
# -----------------------------------------------------------------------
def ask_llm(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
You are a website-based assistant.

Answer the user's question using ONLY the website content provided below.
Do not use outside knowledge.
If the answer is not clearly available in the website content, reply:
"I could not find that information on the provided website."

Website Content:
{context}

User Question:
{question}
"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Website Chatbot"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=API_TIMEOUT
        )

        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            if "error" in data:
                raise RuntimeError(f"OpenRouter API error: {data['error']}")
            raise RuntimeError(f"Unexpected API response: {data}")

        message = data["choices"][0].get("message", {})
        content = message.get("content")

        if not content:
            raise RuntimeError(f"No content found in model response: {data}")

        return content.strip()

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to call OpenRouter API: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response from API: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error while generating response: {e}") from e


# --------------------------------------------------------------------------
# MAIN CONSOLE CHATBOT
# --------------------------------------------------------------------------
def main():
    print("Fetching and processing website content...")

    website_text = fetch_website_text(TARGET_URL)
    chunks = chunk_text(website_text)

    if not chunks:
        raise RuntimeError("No text chunks were created from the website content.")

    print("Website processed successfully.")
    print(f"Total extracted characters: {len(website_text)}")
    print(f"Total chunks created: {len(chunks)}")
    print("\nConsole chatbot is ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            print("Bot: Goodbye!")
            break

        if not user_query:
            print("Bot: Please enter a valid question.\n")
            continue

        try:
            relevant_chunks = retrieve_relevant_chunks(user_query, chunks, TOP_K)
            answer = ask_llm(user_query, relevant_chunks)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"\nBot Error: {e}\n")


if __name__ == "__main__":
    main()
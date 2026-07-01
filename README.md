# 🌐 Website Content Chatbot

A lightweight, console-based chatbot that scrapes content from a target website and uses Retrieval-Augmented Generation (RAG) to accurately answer user questions based *only* on the extracted text[cite: 3]. Powered by **BeautifulSoup** and the **OpenRouter API**[cite: 3].

## ✨ Features

* **Automated Web Scraping**: Fetches and parses HTML content from any specified URL using `requests` and `BeautifulSoup4`[cite: 3].
* **Smart Text Processing**: Cleans and splits the website text into manageable chunks for efficient retrieval[cite: 3].
* **Context-Aware Retrieval**: Uses custom mathematical cosine similarity to find the most relevant chunks of text based on the user's query[cite: 3].
* **LLM Integration**: Connects to the OpenRouter API (utilizing the `qwen/qwen3.6-plus:free` model) to generate natural, conversational answers[cite: 3].
* **Hallucination Mitigation**: Strictly instructed to answer questions based *only* on the provided website content. If the answer is not available, it politely informs the user[cite: 3].
* **Interactive Console UI**: A continuous chat loop running entirely in your terminal[cite: 3].

## 🛠️ Prerequisites

Before you begin, ensure you have Python installed on your machine. You will also need an API key from [OpenRouter](https://openrouter.ai/).

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Install the required dependencies
The script relies on standard web scraping and HTTP libraries. Install them via pip[cite: 3]:
```bash
pip install requests beautifulsoup4
```

### 3. Configure the script
Open `website_chatbot.py` in your text editor and update the `CONFIGURATION` section at the top of the file[cite: 3]:
* `TARGET_URL`: The website you want the chatbot to read (Default is set to `"https://botpenguin.com/"`)[cite: 3].
* `OPENROUTER_API_KEY`: Paste your OpenRouter API key here[cite: 3]. *(Note: Do not commit your actual API key to GitHub!)*

## 💻 Usage

Run the script from your terminal[cite: 3]:

```bash
python website_chatbot.py
```

### Example Interaction

```text
Fetching and processing website content...
Website processed successfully.
Total extracted characters: 15420
Total chunks created: 35

Console chatbot is ready. Type 'exit' to quit.

You: What services does the company offer?
Bot: Based on the website, the company offers AI chatbot development, WhatsApp automation, and live chat support integration.

You: exit
Bot: Goodbye!
```

## 🏗️ How It Works (The Architecture)

1. **Extraction**: `fetch_website_text()` grabs the raw HTML and strips away scripts, styles, and tags to leave clean text[cite: 3].
2. **Chunking**: `chunk_text()` breaks the massive text block into smaller, semantic chunks of 800 characters[cite: 3].
3. **Retrieval**: `retrieve_relevant_chunks()` tokenizes the text and compares the user's query against the chunks using dot-product cosine similarity to find the top 3 highest matches[cite: 3].
4. **Generation**: `ask_llm()` passes the relevant chunks and the user's prompt to the OpenRouter LLM to synthesize a final, accurate response[cite: 3].

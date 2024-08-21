# app.py

from flask import Flask, request, jsonify

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_objectbox.vectorstores import ObjectBox
from langchain.prompts import ChatPromptTemplate


# Initialize embeddings and chat model
EMBEDDING = OllamaEmbeddings(model="mxbai-embed-large")
CHAT_MODEL = ChatOllama(model='tinyllama', embedding=EMBEDDING)

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize ObjectBox with embeddings
objectbox = ObjectBox(embedding=EMBEDDING, embedding_dimensions=1024)

# Define vector search function
def vector_search_db(query_text):
    query_embedding = EMBEDDING.embed_query(query_text)
    db_results = objectbox.similarity_search_by_vector(query_embedding, k=6)
    return db_results

# Define prompt generation function
def generate_prompt(db_results, query_text):
    context_text = "\n\n---\n\n".join([doc.page_content for doc in db_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

# Define response generation function
def generate_response(prompt):
    return CHAT_MODEL.invoke(prompt)

# Define response formatting function
def format_response(response_text, db_results):
    sources = set([doc.metadata.get("source", None) for doc in db_results])
    sources = [source.split("/", 1)[-1].rsplit(".", 1)[0] for source in sources if source is not None]
    return f"Response: {response_text}\nSources: {sources}"

# Initialize Flask app
app = Flask(__name__)

# Define query route
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    db_results = vector_search_db(query_text)
    prompt = generate_prompt(db_results, query_text)
    response_text = generate_response(prompt)
    formatted_response = format_response(response_text, db_results)
    
    return jsonify({"response": formatted_response})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

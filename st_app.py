import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_objectbox.vectorstores import ObjectBox
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Initialize embeddings and chat model
EMBEDDING = OllamaEmbeddings(model="mxbai-embed-large")
CHAT_MODEL = ChatOllama(model='llama3.1', embedding=EMBEDDING)

# Prompt templates for English and Spanish
PROMPT_TEMPLATE_EN = """
Answer the question based only on the following context:

{context}

---

{question}

---

If you cannot answer the question extremely rigorously, you can just helpfully suggest that the user read the source material.
"""

PROMPT_TEMPLATE_ES = """
Responda la pregunta basándose únicamente en el siguiente contexto:

{context}

---

{question}

---

Si no puede responder la pregunta de manera extremadamente rigurosa, simplemente puede sugerir amablemente que el usuario lea el material fuente.
"""

TRANSLATE_EN_ES = "translate English to Spanish:\n{text}"
TRANSLATE_ES_EN = "translate Spanish to English:\n{text}"

# Initialize ObjectBox
def init_objectbox():
    if 'objectbox' not in st.session_state:
        st.session_state.objectbox = ObjectBox(embedding=EMBEDDING, embedding_dimensions=1024, db_directory="objectbox")

init_objectbox()

# Function to handle language translation
def translate_text(text, target_language):
    if target_language == "Spanish":
        return CHAT_MODEL.invoke(TRANSLATE_EN_ES.format(text=text))
    elif target_language == "English":
        return CHAT_MODEL.invoke(TRANSLATE_ES_EN.format(text=text))
    return text

# Define vector search function
def vector_search_db(query_text, language):
    if language == "Spanish":
        query_text = translate_text(query_text, "Spanish")
    query_embedding = EMBEDDING.embed_query(query_text)
    db_results = st.session_state.objectbox.similarity_search_by_vector(query_embedding, k=3)
    return db_results

# Define prompt generation function
def generate_prompt(db_results, query_text, language):
    context_text = "\n\n---\n\n".join([doc.page_content for doc in db_results])
    
    if language == "Spanish":
        prompt_template = PROMPT_TEMPLATE_ES
    else:
        prompt_template = PROMPT_TEMPLATE_EN
    
    prompt = prompt_template.format(context=context_text, question=query_text)
    if language == "Spanish":
        prompt = translate_text(prompt, "English")
    
    return prompt

# Language selection
language = st.radio("Select Language / Seleccione Idioma", ("English", "Spanish"))

# Function to get response
def get_response(user_query, chat_history, language):
    db_results = vector_search_db(user_query, language)
    prompt = generate_prompt(db_results, user_query, language)
    
    chain = prompt | CHAT_MODEL | StrOutputParser()
    
    response_text = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })
    
    sources = set([doc.metadata.get("source", None) for doc in db_results])
    sources = [source.split("/", 1)[-1].rsplit(".", 1)[0] for source in sources if source is not None]
    
    return f"{response_text}\nSources: {sources}"

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?" if language == "English" else "Hola, soy un bot. ¿Cómo puedo ayudarte?"),
    ]

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Type your message here..." if language == "English" else "Escriba su mensaje aquí...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history, language)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))

import sqlite3
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="Mental Health Chatbot: Ask Me About Mental Health")

# Sidebar contents
st.sidebar.write(
    '<p style="color:grey;"> <b> Question History </b>', unsafe_allow_html=True
)

with open("question-history.txt", "r") as f:
    for line in f:
        st.sidebar.write(line)

# Generate empty lists for generated and past.
## generated stores AI generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if "past" not in st.session_state:
    st.session_state["past"] = [""]

# Layout of input/response containers
input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    conn = sqlite3.connect(
        "/Users/nityabhat/Desktop/cmu/ai_chatbot_research/p0/frontend-chatbot/chatbot-react-app/src/keywords_database.db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT url, keywords, summary FROM website_keywords")
    rows = cursor.fetchall()

    nlp = spacy.load("en_core_web_sm")
    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: [token.lemma_ for token in nlp(text)],
        stop_words="english",
    )

    keyword_texts = [row[1] for row in rows]
    tfidf_matrix = vectorizer.fit_transform(keyword_texts)

    user_input_lemmas = [token.lemma_ for token in nlp(user_input)]
    user_input_vector = vectorizer.transform([" ".join(user_input_lemmas)])

    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix)[0]
    similarity_ranking = sorted(
        enumerate(similarity_scores), key=lambda x: x[1], reverse=True
    )[:3]

    resources = []

    for rank, (index, score) in enumerate(similarity_ranking):
        resource_id = rows[index][0]
        resource_summary = rows[index][2]
        resources.append([resource_id, resource_summary])

    buffer = ""
    for rank, (index, score) in enumerate(similarity_ranking):
        resource_id = rows[index][0]
        resource_summary = rows[index][2]
        if rank == 0:
            buffer += f"Here are some resources that you may find useful. The most relevant resource is \n{resource_id}\n"
        elif rank == 1:
            buffer += f"In addition to the above resource, here are two other resources that may help you: \n{resource_id}\n"
        else:
            buffer += f"{resource_id}\n"

        buffer += f"Summary: {resource_summary}\n\n"

    return str(buffer)


with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)

        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

with open("question-history.txt", "a") as f:
    f.write(user_input)
    f.write("\n")

st.sidebar.write(user_input)

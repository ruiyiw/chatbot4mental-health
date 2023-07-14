import sqlite3
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def get_chatbot_response(user_input):
    conn = sqlite3.connect("keywords_database.db")
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

    return resources


st.title("Ask Me About Mental Health!")
question = st.text_area(
    "Hi! How are you doing today? You may enter your question here:"
)  # taking query
st.sidebar.write(
    '<p style="color:grey;"> <b> Question History </b>', unsafe_allow_html=True
)

with open("question-history.txt", "r") as f:
    for line in f:
        st.sidebar.write(line)

if st.button("Get Response"):
    placeholder = st.empty()
    resource_list = get_chatbot_response(question)
    placeholder.text("Fetching information and generating summaries...")

    st.write(
        '<p style="color:red;"> <b> Here is a brief summary of what I found in my knowledgebase: </b>',
        unsafe_allow_html=True,
    )
    st.write(
        f"<p> <b> {resource_list[0][1]} </b>",
        unsafe_allow_html=True,
    )
    st.write(
        '<p style="color:green;"> <i> Further Reading: </i>',
        unsafe_allow_html=True,
    )

    st.write(resource_list[0][0])
    st.write("---")
    st.write(
        '<p style="color:red;"> <b> Here are some additional resources to help you: </b>',
        unsafe_allow_html=True,
    )
    for i in range(1, 3):
        with st.expander(f"**Resource {i}:**"):
            st.write(
                '<p style="color:green;"> <i> Document Summary: </i>',
                unsafe_allow_html=True,
            )
            st.write(resource_list[i][1])
            st.write(
                '<p style="color:green;"> <i> Further Reading: </i>',
                unsafe_allow_html=True,
            )
            st.write(resource_list[i][0])

    placeholder.empty()

    with open("question-history.txt", "a") as f:
        f.write(question)
        f.write("\n")

    st.sidebar.write(question)

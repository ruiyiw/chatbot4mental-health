
import streamlit as st
import pandas as pd

df = pd.read_csv("pdf-collection.csv", index_col=False)
df.head()

from haystack import Document

# Use data to initialize Document objects
titles = list(df["title"].values)
sources = list(df["source"].values)
texts = list(df["text"].values)
documents = []
for title, source, text in zip(titles, sources, texts):
    documents.append(Document(content=text, meta={"pdf_name": title, "url": source}))

from haystack.document_stores import FAISSDocumentStore
import os

if os.path.exists("faiss_storage.db"):
  document_store = FAISSDocumentStore.load("faiss_storage.db")
else:
  document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

from haystack.nodes import DensePassageRetriever, OpenAIAnswerGenerator

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)

generator = OpenAIAnswerGenerator(
    api_key="", # TODO: enter your OpenAI API key. See here: https://platform.openai.com/account/api-keys
    model="text-davinci-003",
    # model='text-davinci-edit-001',
    max_tokens=128,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    top_k=3,
    temperature=0.9
)

if not os.path.exists("faiss_storage.db"):
  # Delete existing documents in documents store
  document_store.delete_documents()

  # Write documents to document store
  document_store.write_documents(documents)

  # Add documents embeddings to index
  document_store.update_embeddings(retriever=retriever)

  document_store.save("faiss_storage.db")

from haystack.pipelines import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)


st.title("Ask Me Anything About Mental Health 🧐")
question = st.text_area("Hi 😊! How are you doing today? You may enter your question here:") # taking query
question_history = []
st.sidebar.write('<p style="color:grey;"> <b> Question History </b>', unsafe_allow_html=True)

with open("question-history.txt", "r") as f:
    for line in f:
        st.sidebar.write(line)

if st.button("Get Response"):
    placeholder = st.empty()
    placeholder.text("Fetching information and generating summary...")
    params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 3}}
    prediction = pipe.run(query=question, params=params)
    
    st.write('<p style="color:orange;"> <b> Here is a brief summary of what I found in my knowledgebase: </b>', unsafe_allow_html=True)
    st.write(f'<p> <b> {prediction["answers"][0].answer} </b>', unsafe_allow_html=True)
    st.write('---')
    st.write('<p style="color:orange;"> <b> If you wanna see more, here are some additional resources: </b>', unsafe_allow_html=True)
    for i in range(3):
        with st.expander(f'**Resource {i+1}:**'):
            st.write('<p style="color:blue;"> <i> Supporting Document: </i>', unsafe_allow_html=True)
            st.write(prediction["answers"][0].meta["content"][i])
            st.write('<p style="color:blue;"> <i> Further Reading: </i>', unsafe_allow_html=True)
            st.write(prediction["answers"][0].meta["doc_metas"][i]["url"])

    placeholder.empty()
    
    with open("question-history.txt", "a") as f:
       f.write(question)
       f.write('\n')
    
    st.sidebar.write(question)


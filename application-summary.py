import streamlit as st
from streamlit_chat import message
from langchain_community.chat_models import AzureChatOpenAI
import os
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pytube import YouTube
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from openai import AzureOpenAI

from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o", api_version="2024-05-01-preview", temperature=0
)


def download_transcript(input_url):
    tmpdir = "/app/"
    # if not os.path.exists("transcription.txt"):
    youtube = YouTube(input_url)
    audio = youtube.streams.filter(only_audio=True).first()

    audio_test_file = audio.download(output_path=tmpdir)

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    deployment_id = "whisper"  # This will correspond to the custom name you chose for your deployment when you deployed a model."

    transcription = client.audio.transcriptions.create(
        file=open(audio_test_file, "rb"), model=deployment_id
    )
    with open("/app/transcription.txt", "w") as file:
        file.write(transcription.text.strip())

    loader = TextLoader("/app/transcription.txt")
    text_documents = loader.load()

    template = """
    You are an AI assistant that helps people find information.
    Please respond from the video transcript provided as input.
    If you can't answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    Answer here:"""

    prompt = ChatPromptTemplate.from_template(template)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)

    embeddings = AzureOpenAIEmbeddings()
    vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)

    chain = (
        {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain


def load_chain():
    prompt_template = """You are an AI assistant that helps people find information. Please respond based on context data.

    {context}

    Question: {question}
    Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    loader = TextLoader("/app/transcription.txt")
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)
    embeddings = AzureOpenAIEmbeddings()
    vector_store = DocArrayInMemorySearch.from_documents(documents, embeddings)

    int_chain = ConversationalRetrievalChain.from_llm(
        llm=AzureChatOpenAI(
            azure_deployment="gpt-4o", api_version="2024-05-01-preview", temperature=0
        ),
        memory=memory,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return int_chain


st.title("Video Summarization App")
st.write("App summarizes the given any YT video")

with st.form("form"):
    input_url = st.text_area("Enter url:")
    submitted = st.form_submit_button("Submit")
    if not input_url.startswith("https://youtube.com/"):
        st.warning("Please enter valid youtube url .. I am still learning...", icon="⚠")
    if submitted:
        chain = download_transcript(input_url)
        st.info(chain.invoke("Summarize the content").content.format())
# if chain:
st.write("Welcome to Video summarization App! Ask me anything from the video")

int_chain = load_chain()

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = int_chain.run(question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

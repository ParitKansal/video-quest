import streamlit as st
from streamlit_chat import message
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)


def load_chain():
    prompt_template = """You are an AI assistant that helps people find information. Please respond based on context data.

    {context}

    Question: {question}
    Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever = AzureCognitiveSearchRetriever(content_key="chunk", top_k=10)

    chain = ConversationalRetrievalChain.from_llm(
        llm=AzureChatOpenAI(azure_deployment="gpt-4o", api_version="2024-05-01-preview",
    temperature=0),
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return chain


chain = load_chain()

st.set_page_config(page_title="videoscribe", page_icon=":robot:")
st.header("videoquest")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

st.write("Welcome to Videoquest! Ask me anything about videoscribe.")
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
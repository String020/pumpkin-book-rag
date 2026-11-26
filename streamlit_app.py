import streamlit as st
from openai import OpenAIError

from retrieval import SetupError, load_qa_chain


@st.cache_resource(show_spinner=False)
def get_qa_chain():
    return load_qa_chain()


def gen_response(chain, question, chat_history):
    for result in chain.stream({"input": question, "chat_history": chat_history}):
        if "answer" in result:
            yield result["answer"]


st.title("Machine Learning Assistant")

try:
    qa_chain = get_qa_chain()
except SetupError as error:
    st.info(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("请输入你的问题")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(gen_response(qa_chain, question, st.session_state.messages))
        except OpenAIError:
            st.error("OpenAI could not complete the request. Check your API key and model access.")
            st.stop()

    st.session_state.messages.append(("user", question))
    st.session_state.messages.append(("assistant", answer))

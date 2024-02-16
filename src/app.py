import streamlit as st
from streamlit_chat import message
from PIL import Image
from utils.load_config import LoadConfig
from utils.app_utils import load_data, RAG, delete_data
import subprocess
import os

APPCFG = LoadConfig()

# ===================================
# Setting page title and header
# ===================================
im = Image.open("images/bear_bears.jpg")
#os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

st.set_page_config(page_title="RAG-Maestro", page_icon=im, layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>RAG-Maestro (Scientific Assistant)</h1>",
    unsafe_allow_html=True,
)
st.divider()
st.markdown(
        "<center><i>RAG-Maestro is an up-to-date LLM assistant designed to provide clear and concise explanations of scientific concepts <b>and relevant papers</b>. As a Q&A bot, it does not keep track of your conversation and will treat each input independently.  Do not hesitate to clear the conversation once in a while! Hoping that RAG-Maestro will help get quick answers and expand your scientific knowledge.</center>",
        unsafe_allow_html=True,
    )
st.divider()

#Session State is a way to share variables between reruns, for each user session
# ===================================
# Initialise session state variables
# ===================================
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
# ==================================
# Sidebar:
# ==================================
counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center;'>Ask anything you need to brush up on!</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><b>Example: </b></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>What is GPT4?</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>Explain me Mixture of Models (MoE)</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>How does RAG works?</i></center>",
        unsafe_allow_html=True,
    )
    # st.sidebar.title("An agent that read and summarizethe the news for you")
    st.sidebar.image("images/bare_bears.jpg", use_column_width=True)
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    st.markdown(
    "<a style='display: block; text-align: center;' href='https://github.com/sahithyaswaminathan' target='_blank'> Sahithya Swaminathan</a>",
    unsafe_allow_html=True,
)
    
# ==================================
# Reset everything (Clear button)
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    delete_data()

response_container = st.container()  # container for message display

if query := st.chat_input(
    "What do you need to know? I will explain it and point you out interesting readings."
):
    st.session_state["past"].append(query) #appending past queries
    try:
        with st.spinner('Reading the best papers...'):
            command = 'python'
            script_path = 'src/utils/arxiv_scrapper.py'
            args = [command, script_path, '--query', "'{query}'", '--num_result', '{APPCFG.articles_to_search}']
            process = subprocess.Popen(
                args=args
            )
            out, err = process.communicate()
            errcode = process.returncode
        
        with st.spinner("Reading them..."):
            data = load_data()
            index = RAG(APPCFG, _docs=data)
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                verbose=True,
                similarity_top_k=APPCFG.similarity_top_k,
            )
        with st.spinner("Thinking..."):
            response = query_engine.query(query + APPCFG.llm_format_output)
        
        st.session_state["generated"].append(response.response)
        del index
        del query_engine

        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True)

                message(st.session_state["generated"][i], is_user=False)
        
    except Exception as e:
        print(e)
        st.session_state["generated"].append(
            "An error occured with the paper search, please modify your query."
        )

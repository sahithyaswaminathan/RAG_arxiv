from typing import List
import streamlit as st
import os, shutil
import accelerate
import torch
import safetensors
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import HuggingFaceLLM, OpenLLMAPI
from llama_index import SimpleDirectoryReader
from transformers import AutoModel, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_data():
    reader = SimpleDirectoryReader(input_dir="src/data", recursive=True)
    docs = reader.load_data()
    return docs

def RAG(_config, _docs):
    print('service context start')
    system_prompt = """
    As a chatbot, your goal is to respond to the user's question respectfully and concisely.\
  You will receive the user's new query, along with 3 articles from the web search result for that query.\
  Answer the user with the most relevant information. After answering, cite your sources and provide the url.
    """
    huggingface_llm = HuggingFaceLLM(
            # model='StabilityAI/stablelm-tuned-alpha-3b',
            # #temperature=_config.temperature,
            # max_new_tokens=256,
            # system_prompt=system_prompt,
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            system_prompt=system_prompt,
            #query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
            model_name="StabilityAI/stablelm-tuned-alpha-3b",
            device_map="auto",
            stopping_ids=[50278, 50279, 50277, 1, 0],
            tokenizer_kwargs={"max_length": 4096},
        )
    if safetensors.is_available():
        safetensors.set_offload_directory("your_offload_folder_name")
    else:
        raise Exception("safetensors is not installed. Please install safetensors to handle offloaded weights.")

    service_context = ServiceContext.from_defaults( 
        llm=huggingface_llm,
        chunk_size=1024,
        embed_model='local'
    )

    # model_name = "StabilityAI/stablelm-tuned-alpha-3b"
    # tokenizer_name = model_name  # usually the same as model_name

    # model = AutoModel.from_pretrained(model_name, offload="offload")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # embedding = HuggingFaceEmbedding(
    #     model_name=model_name,
    #     tokenizer_name=tokenizer_name,
    #     model=model,
    #     tokenizer=tokenizer,
    # )
    print('service context done')
    index = VectorStoreIndex.from_documents(_docs, service_context=service_context)
    return index

def delete_data():
    print("Cleaning the data folder")
    folder = "src/data"
    for filename in os.listdir(folder):
        if filename != ".gitignore":
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
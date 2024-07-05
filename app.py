from langchain_community.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.prompts import PromptTemplate
import torch
import subprocess
from typing import List

# We use all-MiniLM-L6-v2 as embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings():
  def embed_query(self, text: str) -> List[float]:
    return model.encode(text).tolist()

embeddings = LocalEmbeddings()

# Connect to Epsilla as knowledge base.
client = vectordb.Client()
vector_store = Epsilla(
  client,
  embeddings,
  db_path="/tmp/demodb",
  db_name="DemoDB",
)
vector_store.use_collection("DemoCollection")

st.image("https://hatchworks.com/wp-content/uploads/2022/03/HatchWorks-Logo-No-Tag-Horizontal-Color.png", width=500)
# The 1st welcome message
st.title("Sagemaker wizard ")
if "messages" not in st.session_state:
  st.session_state["messages"] = [{"role": "assistant", "content": "I'm here to answer any question you might have related to AWS Sagemaker!"}]

# A fixture of the chat history
for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

# Answer user question upon receiving
if question := st.chat_input():
  st.session_state.messages.append({"role": "user", "content": question})

  # Here we find 5 pieces of chunks that appear to  be more relevant to answer the question using similarity_search 
  context = '\n'.join(map(lambda doc: doc.page_content, vector_store.similarity_search(question, k = 1)))

  st.chat_message("user").write(question)

  # Now we use prompt engineering to ingest the most relevant pieces of chunks from our epsila knowledge database into the prompt.
  template = """
    Answer the Question based on the given Context. Try to understand the Context and rephrase them.
    Please don't make things up or say things not mentioned in the Context. Provide short and clear answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

  #print(template)

  prompt = PromptTemplate(template=template, input_variables=["question", "context"])

  # Callbacks support token-wise streaming
  callbacks = CallbackManager([StreamingStdOutCallbackHandler()])

  n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
  n_ctx = 3584
  n_batch = 3584  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

  llm = LlamaCpp(
    model_path="models/mistral-7b-openorca.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    n_batch=n_batch,
    callback_manager=callbacks,
    verbose=True,  # Verbose is required to pass to the callback manager
  )

  # Call the local LLM and wait for the generation to finish. 
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  content = llm_chain.run(context=context, question=question)

  # Append the response
  msg = { 'role': 'assistant', 'content': content }
  st.session_state.messages.append(msg)
  st.chat_message("assistant").write(msg['content'])
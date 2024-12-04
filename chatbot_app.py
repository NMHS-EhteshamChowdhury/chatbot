import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS

def process_input(question):
    #model_local = ChatOllama(model="mistral")
    model_local = ChatOllama(model="my_llama2")
    
    docs_paths = ["./National_Efficient_Price_Determination_2022_23.pdf", 
                  "./National_Efficient_Price_Determination_2023_24.pdf"]

    all_doc_splits = []

    for path in docs_paths:
        loader = PyPDFLoader(path)
        doc_splits = loader.load_and_split()
        all_doc_splits.extend(doc_splits)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100) 
    chunk_doc_splits = text_splitter.split_documents(all_doc_splits)

    # embeddings = OllamaEmbeddings(model='nomic')
    # db = FAISS.from_documents(chunk_doc_splits, embeddings)

    # retriever = db.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={'k': 4}
    # )

    vectorstore = Chroma.from_documents(
        documents=chunk_doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic'),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}. If it's out of your ability to answer the question, don't answer with falsification of facts and apologize for the inconvenience.
    Provide a refernece of in which file which page you found the answer at the end.
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)
 
# Title for your app
st.title('Chatbot Interface')
 
# Text input for the question
question = st.text_input("Ask a question:")
 
# Button to submit the question
if st.button('Submit'):
    if question:
        # Process the input question through your chatbot's pipeline
        response = process_input(question)
        # Display the response
        st.text_area("Response:", value=response, height=200, max_chars=None, key=None)
    else:
        st.write("Please enter a question.")
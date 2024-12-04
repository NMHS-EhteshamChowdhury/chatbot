import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain import PromptTemplate
import chainlit as cl



llm_local = ChatOllama(model="my_llama2")

@cl.on_chat_start
async def on_chat_start():

    files = None # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180, 
        ).send()

    file = files[0] # Get the first uploaded file

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    pdf =  PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    # Create a Chroma vector store
    
    embeddings = OllamaEmbeddings(model='nomic')
    #PY try diff model
    #embeddings = OllamaEmbeddings(model='nomic-embed-text')
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    retriever = docsearch.as_retriever()

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key ="question",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create template
    template = """[INST]
    Please provide an answer to the latest question only, using the context provided below.
    Do not consider previous conversations or external information beyond the current context.
    Context: {context}
    Chat History: {chat_history}
    Focus on this qustion solely: {question}
    Your response should be strictly informed by the context given above.
    If the question cannot be answered accurately with the given context and history,
    please abstain from making unsupported statements and kindly note the limitation.
    [/INST]"""
    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

    # Create a chain that uses Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    #store the chain in user session
    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message: cl.Message):
        
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    # return results
    await cl.Message(content=answer, elements=text_elements).send()
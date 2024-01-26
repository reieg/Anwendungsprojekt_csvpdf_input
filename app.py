from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings#importing from langchain but come from openai
from langchain.vectorstores import FAISS#faiss= facebook ai similarity search library
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback#optional:check how much money the search spends

# importing pdf files:
def extract_text_from_pdf(pdf_files):
    combined_text = ""
    for pdf in pdf_files:#test if pdf file exists
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:#reader does not allow to loop upload multiple pages at once
            text += page.extract_text()#takes the string text out of the page
        combined_text += text + "\n"  # Concatenate text with a newline separator
    return combined_text

# importing csv files --> funktioniert nicht für alle layouts bislang
def extract_text_from_csv(csv_files):
    combined_text = ""
    for csv in csv_files:
        df = pd.read_csv(csv)
        text = df.to_string(index=False)
        combined_text += text + "\n"  # Concatenate text with a newline separator
    return combined_text


def main():
    load_dotenv()#zugriff auf .env bzw. api key
    st.set_page_config(page_title="Ask your Documents")#User Interface with streamlit
    st.header("Ask your Documents 💬")
    
    # upload multiple files - PDF and CSV
    pdf_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)#hier muss Datenbank angebunden werden
    csv_files = st.file_uploader("Upload your CSVs", type="csv", accept_multiple_files=True)#hier muss Datenbank angebunden werden
    
    # extract text from uploaded PDFs and CSVs and concatenate
    combined_text = ""
    if pdf_files is not None:
        combined_text += extract_text_from_pdf(pdf_files)
    if csv_files is not None:
        combined_text += extract_text_from_csv(csv_files)
    
    # process the combined text if at least one file was uploaded
    if combined_text:
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,#1000 characters for each chunk --> maybe test/optimize this
            chunk_overlap=200,
            length_function=len#to measure the chunk length
        )
        chunks = text_splitter.split_text(combined_text)#to eventually feed relevant chunks only to the model, passing the whoole text variable
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)#semantik search in the knowledge base --> combining chunks and embeddings
        
        # show user input
        user_question = st.text_input("Ask a question about the uploaded documents:")#create text box
 
        if user_question:
        
            from langchain.memory import ConversationBufferMemory
            from langchain.chains import ConversationalRetrievalChain
            from langchain.chat_models import ChatOpenAI

        
            # GPT-3.5 Modell initialisieren
            llm = ChatOpenAI(temperature=0.2)#hatte hier schonmal probiert das model zu definieren, was nicht geklappt hat: llm = ChatOpenAI(model_name="text-davinci-003",temperature=0.2)
            # Memory initialisieren
            memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
            # Retriever initialisieren
            retriever = knowledge_base.as_retriever() # Chunks aus erstellter FAISS-DB abfragen
            

            # Neu: Individuelle System Message ergänzen:
            
            # zusätzliche Imports:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            # System Message 1 definieren: -> chat_history + neue Frage
            _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
            standalone question without changing the content in given question.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
            condense_question_prompt_template = PromptTemplate.from_template(_template)

            # System Message 2 definieren: context + foolow up question
            prompt_template = """Ich will, dass du als Klausureinsicht Helfer agierst und mit Studenten redest. Dein Name ist Klausureinsicht-Assistent. Die Antwort der Studenten ist bereits in der Datenbank enthalten. You are helpful information giving QA System and make sure you don't answer anything 
            not related to following context. You are always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 

            {context}

            Question: {question}
            Helpful Answer:"""

            qa_prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
            doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
            crchain = ConversationalRetrievalChain(
                retriever=retriever,
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
                memory=memory,

            )

            # Initialisation des Session-States:
            if "crchain" not in st.session_state:
                st.session_state.crchain = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = None

            # crchain mit Hilfe von Streamlit‘s “session_state“ persistent speichern
            if st.session_state.chat_history == None:
                st.session_state.crchain = crchain
            response = st.session_state.crchain(user_question)  # persistent gespeicherte ConversationalRetrievalChain unter Übergabe der Userfrage aufrufen
            
            # chat_history mit Hilfe von Streamlit‘s “session_state“ persistent speichern
            st.session_state.chat_history = response['chat_history']
            
            # Ausgabe der Chat-History
            if st.session_state.chat_history is not None:
                st.write("Chat History:")
                chat_history = st.session_state.chat_history
                for idx in range(0, len(chat_history), 2):
                    st.text(f"Me: {chat_history[idx]}")
                    if idx + 1 < len(chat_history):
                        st.text(f"AI: {chat_history[idx + 1]}")


if __name__ == '__main__':
    main()
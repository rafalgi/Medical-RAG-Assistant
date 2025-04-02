from langchain_ollama.llms import OllamaLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List 
import bs4
import re
import streamlit as st

from src.utils import load_documents_from_urls, split_documents, create_vectorstore
from src.prompts import template


llm = OllamaLLM(model="deepseek-r1:7b", temperature=0.1, max_tokens=512)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")


web_paths = ['https://medlineplus.gov/ency/article/000033.htm',
             'https://pmc.ncbi.nlm.nih.gov/articles/PMC10503338/',
             'https://www.ncbi.nlm.nih.gov/books/NBK537235/',
             'https://medicaljournalssweden.se/actadv/article/view/11592/19144',
             'https://emedicine.medscape.com/article/769067-overview?form=fpf',
             'https://www.mayoclinic.org/first-aid/first-aid-insect-bites/basics/art-20056593',
             'https://www.medicalnewstoday.com/articles/174229#reactions',
             'https://www.aafp.org/pubs/afp/issues/2022/0800/arthropod-bites-stings.html',
             'https://wwwnc.cdc.gov/travel/page/avoid-bug-bites']

all_documents = load_documents_from_urls(web_paths)
all_splits = split_documents(all_documents)
vectorstore = create_vectorstore(all_splits)

retriever = vectorstore.as_retriever()


def clean_text(text):
    '''
    Remove the <think> tags and their content from the text.
    '''

    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def reason(text):
    '''
    Extract the reasoning from the text by searching for the <think> tags.
    '''

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else "No reasoning found."

def setup_rag_chain(question) -> str:
    '''
    Set up the RAG chain with the LLM and retriever, and format the input data.
    '''

    retrieved_context = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in retrieved_context])
    
    prompt = ChatPromptTemplate.from_template(template)
    input_data = prompt.format(question=question, context=context)
    
    chain = (
        llm
        | StrOutputParser()
    )
    response = chain.invoke(input_data)
    return response





def main():
    '''
    Main function to run the Streamlit app.
    '''
    
    st.title("🧑🏼‍⚕️Medical RAG Assistant")
    st.subheader("Ask me anything about the medical field!")

    # Initialize session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []  # Ensure history is initialized as an empty list

    question = st.chat_input()

    if question:
        # Add user input to history
        st.session_state.history.append({"role": "user", "content": question})

        response = setup_rag_chain(question)
        
        # Process the response
        reasoning = reason(response)
        answer = clean_text(response)
        sources = "\n".join([doc.metadata["source"] for doc in retriever.get_relevant_documents(question)])

        # Add assistant response to history
        st.session_state.history.append({
            "role": "assistant",
            "content": f"**Thinking:** {reasoning}\n\n**Answer:** {answer}\n\n**Source:** {sources}"
        })

    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

if __name__ == "__main__":
    main()


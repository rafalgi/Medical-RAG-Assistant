from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List 
import bs4


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


def load_documents_from_urls(urls: List[str]) -> List[str]:
    """
    Loads documents from a list of urls using WebBaseLoader.
    """

    loader = WebBaseLoader(
        web_paths = web_paths,
        bs_kwargs = dict(
            parse_only = bs4.SoupStrainer(
                class_ = ["article", "main", "content", "text"],
            )
        )
    )

    documents = loader.load()

    return documents

def split_documents(documents: List[str]) -> List[str]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter for easier 
    processing and embedding.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        add_start_index = True # oznacza, że każdemu wygenerowanemu fragmentowi tekstu zostanie przypisany indeks początkowy 
    )

    all_splits = text_splitter.split_documents(documents)

    return all_splits


def create_vectorstore(all_splits: List[str]) -> FAISS:
    """
    Creates a vector store using FAISS for efficient similarity search.
    """

    vectorstore = FAISS.from_documents(all_splits, embeddings)

    return vectorstore

 
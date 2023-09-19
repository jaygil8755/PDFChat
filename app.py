package__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.callbacks.base import BaseCallbackHandler

import tempfile

import os


# set page config
st.set_page_config(page_title='PDFChat', layout="centered")
st.subheader('[미니프로젝트] 나만의 PDF Q&A 인공지능')
st.markdown('### :book::sunglasses: Langchain 활용 PDF 챗봇 서비스 - `langchain`, `openai`')

# 사용자로부터 PDF 파일을 받기
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

# 사용자로부터 OPEN AI KEY를 받기
st.write("OenA API 키를 입력해주세요.")
OPENAI_API_KEY = st.text_input('OPEN_AI_API_KEY', type='password')

# 1. 파일을 페이지단위로 나눠 글자 추출하기
def pdf_to_document (uploaded_file ):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open (temp_filepath, 'wb') as f:
               f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
    
# 2. 텍스트를 Chunk 단위로 split하기

if uploaded_file is not None:
    pages = pdf_to_document (uploaded_file )
    
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)
    
    texts = text_splitter.split_documents(pages)
                       
#3. 임베딩하기
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#4. 벡터DB에 로딩하기
    # db = Chroma.from_documents(texts, embeddings_model)
    db = FAISS.from_documents(texts, embeddings_model)

#5. 질문을 계속 받아줄 수 있는 핸들러 만들기
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
           
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token 
            self.container.markdown(self.text)

#6. 질문하고 답변받기
    st.markdown("#### PDF 내용에 대해 무엇이든 물어보세요")
    question = st.text_input("질문을 적으세요")
                       
    if st.button ('제출'):
        with st.spinner("🤖 열심히 작업 중..... "):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(openai_api_key= OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            qa_chain({"query": question})


st.caption("감사합니다. 궁금하신 사항은 jaygil8755@gmail.com으로 문의해주세요")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

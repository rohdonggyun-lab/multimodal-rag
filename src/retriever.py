"""③ 검색·생성: 다중 벡터 검색기, 최종 멀티모달 체인, 비교용 텍스트 전용 RAG.

노트북 Part 3 (Multi_modal_RAG.ipynb)의 함수를 옮긴 모듈.
"""
import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


ID_KEY = "doc_id"
EMBEDDING_MODEL = "text-embedding-3-small"
BASE64_PATTERN = "^[A-Za-z0-9+/]{100,}={0,2}$"


def create_retriever(collection_name="multimodal_rag"):
    """요약은 벡터 DB(Chroma)에, 원본은 docstore에 두는 다중 벡터 검색기를 만든다."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
    )


def add_documents(retriever, doc_summaries, doc_contents):
    """요약은 벡터 DB에, 원본은 docstore에 같은 doc_id로 저장한다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눈다."""
    b64_images = []
    texts_out = []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if re.match(BASE64_PATTERN, doc):
            b64_images.append(doc)
        else:
            texts_out.append(doc)
    return {"images": b64_images, "texts": texts_out}


def is_table(text):
    """VLM으로 만든 표 원본은 '[N페이지 표]' 로 시작한다."""
    return text.startswith("[") and "페이지 표]" in text.split("\n")[0]


def img_prompt_func(data_dict):
    """검색된 원본(이미지 + 텍스트)과 질문을 멀티모달 LLM 입력으로 조립한다."""
    messages = []
    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": (
            "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
            "자료에 없는 내용은 모른다고 답하세요.\n\n"
            f"질문: {data_dict['question']}\n\n"
            f"참고 텍스트:\n{formatted_texts}"
        ),
    })
    return [HumanMessage(content=messages)]


def build_chain(retriever, model=None):
    """질문 → 검색 → 이미지/텍스트 분리 → 멀티모달 LLM 답변 체인을 만든다."""
    model = model or ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def build_text_only_chain(texts, model=None, k=4):
    """Before/After 비교용: 텍스트 청크만 검색하는 RAG 체인을 만든다."""
    model = model or ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    text_only_store = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    text_only_store.add_texts(texts)
    text_only_retriever = text_only_store.as_retriever(search_kwargs={"k": k})

    text_only_prompt = ChatPromptTemplate.from_template(
        "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
        "자료에 없는 내용은 모른다고 답하세요.\n\n"
        "4줄로 답하세요.\n\n"
        "질문: {question}\n\n참고 텍스트:\n{context}"
    )
    return (
        {
            "context": text_only_retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | text_only_prompt
        | model
        | StrOutputParser()
    )

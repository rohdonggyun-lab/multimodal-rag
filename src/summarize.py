"""② 요약: 텍스트·표·이미지를 검색용 요약문으로 바꾼다.

노트북 Part 2 (Multi_modal_RAG.ipynb)의 함수를 옮긴 모듈.
"""
import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image as PILImage


TEXT_PROMPT_KOR = (
    "표와 텍스트를 검색용으로 요약하는 도우미 / 임베딩되어 원본 검색에 쓰임 / "
    "검색에 최적화된 간결한 요약 작성 / 절대 많이 요약하지 말 것 요약률 30% 이하 / "
    "요약할 표 또는 텍스트: {element}"
)

IMAGE_PROMPT_KOR = (
    "이 이미지는 기업 IR 자료(PDF)에서 잘라낸 그림입니다. 검색에 쓰일 요약을 한국어로 작성하세요.\n"
    "1. 이미지 안에 제목이나 머리글이 있으면 그대로 적으세요.\n"
    "2. 차트(막대, 도넛, 선)라면 항목명, 수치, 단위, 기간을 빠짐없이 적으세요.\n"
    "3. 이미지에 보이지 않는 내용이나 용도는 추측하지 마세요.\n"
    "4. 목차, 표지, 로고, 아이콘, 배경·장식 사진처럼 수치나 사실 정보가 없는 이미지는 SKIP 이라고만 답하세요."
)

MIN_SIDE = 150  # 가로나 세로가 이 픽셀보다 작은 이미지는 요약하지 않는다


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트와 표를 검색용 요약으로 바꾼다. summarize_texts=False 면 텍스트는 원문을 그대로 쓴다."""
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    summarize_chain = prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 바꾼다."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Base64 이미지 한 장을 VLM으로 요약한다."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])
    return chat.invoke([message]).content


def generate_img_summaries(path, min_side=MIN_SIDE, verbose=True):
    """폴더 안의 이미지를 요약한다. 작은 이미지와 SKIP 이미지는 원본·요약 모두 제외한다.

    반환: (img_base64_list, image_summaries, kept_files) — 세 리스트는 순서가 같다.
    """
    img_base64_list = []
    image_summaries = []
    kept_files = []
    n_small, n_skip = 0, 0

    for file_name in sorted(os.listdir(path)):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        full_path = os.path.join(path, file_name)
        with PILImage.open(full_path) as im:
            if min(im.size) < min_side:
                n_small += 1
                continue
        img_base64 = encode_image(full_path)
        summary = image_summarize(img_base64, IMAGE_PROMPT_KOR).strip()
        if summary.upper().startswith("SKIP"):
            n_skip += 1
            continue
        img_base64_list.append(img_base64)
        image_summaries.append(summary)
        kept_files.append(file_name)

    if verbose:
        print(f"작은 이미지 제외: {n_small}개 / SKIP 제외: {n_skip}개 / 색인 대상: {len(image_summaries)}개")

    return img_base64_list, image_summaries, kept_files

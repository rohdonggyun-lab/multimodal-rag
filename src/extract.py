"""① 추출: PDF에서 텍스트·이미지를 꺼내고, 표는 VLM으로 보완한다.

노트북 Part 1 (Multi_modal_RAG.ipynb)의 함수를 옮긴 모듈.
"""
import base64
import io

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from pdf2image import convert_from_path, pdfinfo_from_path
from unstructured.partition.pdf import partition_pdf


TABLE_PROMPT = """이 페이지의 모든 표를 마크다운 표로 옮겨 적으세요.
1. 표마다 열 머리글을 왼쪽부터 빠짐없이 적으세요. 병합된 머리글(예: '2025' 아래에 '1Q', '2Q')은 '2025 1Q', '2025 2Q'처럼 하위 열마다 풀어서 쓰세요. 연간 열과 분기 열이 함께 있으면 서로 구분되게 쓰세요.
2. 모든 행이 머리글과 같은 수의 값을 갖게 하세요. 열을 건너뛰거나 합치지 마세요.
3. 행 이름은 보이는 그대로 쓰세요. 비슷한 단어로 바꾸지 마세요.
4. 제목과 단위를 표 앞에 적고, 숫자와 기호(△, %, -, 괄호)는 보이는 그대로 쓰세요. 없는 내용은 지어내지 마세요.
그래프, 사진, 아이콘, 목차, 장식은 표가 아닙니다. 표가 하나도 없으면 NO_TABLE 이라고만 답하세요."""


def extract_pdf_elements(fpath, fname):
    """PDF에서 텍스트 블록과 이미지를 추출한다. 이미지는 fpath + "figures" 에 저장된다.

    fpath: PDF가 있는 폴더 경로 (끝에 경로 구분자 포함)
    fname: PDF 파일 이름
    """
    return partition_pdf(
        filename=fpath + fname,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_output_dir=fpath + "figures",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        languages=["kor", "eng"],
    )


def categorize_elements(raw_pdf_elements):
    """partition_pdf 결과를 텍스트와 표로 나눈다. (이 자료에서는 표가 0개로 나옴)"""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        name = type(element).__name__
        if name in ("Table", "TableChunk"):
            tables.append(str(element))
        elif name == "CompositeElement":
            texts.append(str(element))
    return texts, tables


def extract_tables_with_vlm(pdf_path, dpi=200, model="gpt-4o-mini", verbose=True):
    """PDF의 각 페이지를 이미지로 바꿔 VLM이 표를 마크다운으로 옮기게 한다.

    반환: "[N페이지 표]\\n<마크다운 표>" 형태의 문자열 리스트
    """
    vlm = ChatOpenAI(model=model, temperature=0, max_tokens=3000)
    n_pages = pdfinfo_from_path(pdf_path)["Pages"]
    tables = []

    for p in range(1, n_pages + 1):
        img = convert_from_path(pdf_path, dpi=dpi, first_page=p, last_page=p)[0]
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        msg = HumanMessage(content=[
            {"type": "text", "text": TABLE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])
        result = vlm.invoke([msg]).content.strip()
        result = result.replace("```markdown", "").replace("```", "").strip()

        found = not result.startswith("NO_TABLE")
        if verbose:
            print(f"{p}페이지:", "표 발견" if found else "표 없음")
        if found:
            tables.append(f"[{p}페이지 표]\n{result}")

    return tables


def split_texts(texts, chunk_size=2000, chunk_overlap=400):
    """텍스트 블록을 이어 붙인 뒤 토큰 기준으로 다시 나눈다."""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(" ".join(texts))

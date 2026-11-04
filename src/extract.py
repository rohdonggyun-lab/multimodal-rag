from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    path: 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                # 레이아웃 분석 모델로 정밀 추출 (교재에 없음! 필수 추가)
        extract_images_in_pdf=True,       # PDF에서 이미지를 추출
        infer_table_structure=True,       # 테이블 구조를 추론
        chunking_strategy="by_title",     # 타이틀을 기준으로 텍스트를 블록으로 분할
        max_characters=4000,              # 최대 4000자로 텍스트 블록을 제한
        new_after_n_chars=3800,           # 3800자 이후에 새로운 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하의 텍스트는 결합
        image_output_dir_path=path,       # 이미지가 저장될 경로 설정
    )
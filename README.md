## 설치 및 실행 준비

```bash
git clone <저장소 주소>
cd multimodal-rag

# 가상환경 생성
python -m venv .venv
```

가상환경을 활성화한 뒤 의존성을 설치합니다.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows 명령 프롬프트(cmd)
.venv\Scripts\activate.bat
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
# 또는 고정된 의존성 버전을 사용하려면:
# pip install -r requirements.lock.txt
```

Jupyter에서 사용할 커널을 등록하고 `.env` 파일을 준비합니다.

```bash
python -m ipykernel install --user --name multimodal-rag

# macOS / Linux
cp .env.example .env

# Windows 명령 프롬프트(cmd)
copy .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

`.env` 파일을 열어 필요한 실제 API 키를 입력하세요.

## 프로젝트 구조

```text
multimodal-rag/
├── data/
│   ├── 2Q26 현대무벡스 IR BOOK.pdf
│   └── figures/
├── notebooks/
│   ├── Multi_modal_RAG.ipynb
│   ├── test.ipynb
│   └── test.py
├── src/
│   ├── extract.py
│   ├── retriever.py
│   └── summarize.py
├── .env.example
├── requirements.txt
├── requirements.lock.txt
└── README.md
```

- `data/`: 분석에 사용하는 PDF 문서와 이미지 자료
- `notebooks/`: 실험 및 실행용 노트북과 스크립트
- `src/`: 문서 추출, 검색, 요약 관련 코드
- `.env.example`: 환경 변수 설정 예시
- `requirements.txt`, `requirements.lock.txt`: Python 패키지 의존성 목록


## 다른 팀원에게 환경 전달하기

```
git clone <저장소 주소>
cd multimodal-rag

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt

# 커널 등록
python -m ipykernel install --user --name multimodal-rag

# .env 작성
cp .env.example .env                # 윈도우: copy .env.example .env
# .env 파일을 열어 실제 API 키를 입력
```
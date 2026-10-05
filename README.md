# scope-digest
SCOPE Lab 연구주제 기반 주간 논문 동향 페이지 (LLM 불필요, 무료, 비밀번호 불필요).
OpenAlex(출판 논문) + arXiv(ML 방법론 프리프린트) -> TF-IDF 분류 -> 급상승 키워드 -> 교차 갭 후보 -> `docs/`에 HTML/TXT 생성 -> GitHub Pages.

## 설정
1. 저장소 Settings -> Pages -> Source: `Deploy from a branch`, Branch `main`, folder `/docs`.
2. Settings -> Actions -> General -> Workflow permissions: `Read and write`.
3. Actions 탭 -> `weekly-digest` -> Run workflow 로 첫 호출. 이후 매주 목요일 08:00 KST 자동 실행.
4. 페이지: `https://<계정>.github.io/<저장소>/` (최신호), `/archive.html` (지난 호).
로컬 확인: `pip install -r requirements.txt && python digest.py`

## 무료 LLM (선택)
GitHub Models는 2026-07-30 종료되어 사용하지 않습니다. Google AI Studio(aistudio.google.com/apikey)에서 무료 키를 발급해
저장소 Settings -> Secrets -> Actions -> `GEMINI_API_KEY` 로 등록하세요. 키가 없거나 호출이 실패하면 알고리즘 모드로 발행됩니다.
무료 등급은 입력 데이터가 Google 제품 개선에 쓰일 수 있습니다(공개 논문 초록만 전송). 키 유출 대비로 전용 프로젝트에서 발급하세요.

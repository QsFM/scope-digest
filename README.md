# scope-digest (SCOPE Weekly)
연구주제 기반 주간 논문 리포트. 결과는 `docs/`의 HTML/TXT로 생성되고 GitHub Pages로 읽습니다. 메일·비밀번호 불필요.

## 구조
수집(OpenAlex → 막히면 Crossref, arXiv는 ML 방법론만) → 키워드 일치로 분야 분류 → 급상승 표현·교차 갭 집계 → (선택) Gemini가 서술형 리포트 작성 → docs/ 저장

- ML 방법론 분야: 저널 제한 없음 (+ arXiv 프리프린트)
- 그 외 분야: `config.yaml`의 `journals`(SCIE 저널 ISSN 10개씩)로 한정

## 설정 (1회)
1. Settings → Actions → General → Workflow permissions → Read and write
2. Settings → Secrets and variables → Actions → New repository secret
   - `KHU_API_KEY` : ChatKHU 로그인 → 좌측 하단 API Gateway → API 키 생성 (선택. 호출 크레딧은 본인 계정에서 차감)
   - `GEMINI_API_KEY` : aistudio.google.com/apikey 에서 무료 발급 (서술형 리포트에 필요)
   - `OPENALEX_API_KEY` : openalex.org/settings/api 에서 무료 발급 (수집 안정화)
3. Actions → weekly-digest → Run workflow (요일 무관, 즉시 실행)
4. 첫 실행 후 Settings → Pages → Branch main, 폴더 /docs
스케줄: 매주 목요일 08:00 KST

## 조정 (config.yaml)
- `topics.*.keywords` : 분야별 검색어. `topics.*.journals` : 저널(ISSN). 저널 목록은 직접 고른 것이며 JCR 순위가 아님.
- `window_days`, `min_score`(잡음이 많으면 3), `max_papers_per_topic`
- `llm.models` : 시도할 Gemini 모델 순서

로컬 실행: `pip install -r requirements.txt && python digest.py` (`--dry-run` 이면 preview.html만 생성)

## LLM 순서
`config.yaml`의 `llm.providers`(기본 khu → gemini). KHU 키가 없거나 실패(크레딧 소진 등)하면 Gemini로, 둘 다 안 되면 알고리즘 모드로 발행합니다.
실패 원인은 페이지 상단 안내에 표시됩니다(키 값은 표시되지 않음). ChatKHU 모델은 `llm.khu_models`에서 바꿉니다.
실행 로그의 `khu 크레딧(호출 전/후)` 줄로 1회 사용량을 가늠할 수 있습니다(잔액 조회가 지원될 때).

## 영문판 (v2)
한글 리포트를 만든 뒤 같은 LLM 순서(khu → gemini)로 영어로 번역해 `YYYY-MM-DD-en.html`, `index-en.html`로 발행합니다(호출 1회 추가). 끄려면 `llm.english: false`. 번역이 실패하면 영문판은 알고리즘 집계 페이지로 나오고 원인이 상단에 표시됩니다.

## 암호 보호 (v2)
- 모든 페이지(최신호, 지난 호, txt판, 아카이브)를 AES-256-GCM으로 암호화하고, 브라우저에서 암호를 입력하면 열립니다(PBKDF2-SHA256 60만 회). 탭을 닫기 전까지 같은 탭에서는 다시 묻지 않습니다.
- Secret `PAGE_PASSWORD`(12자 이상) 등록 필수. 없으면 평문 발행을 막기 위해 실행이 중단됩니다. `encrypt: false`로 끌 수 있습니다.
- 첫 실행 때 기존 평문 문서는 자동으로 암호화되고 `.txt`·`docs/archive.json`은 삭제됩니다(`.txt`는 `-txt.html`로 대체, 아카이브 메타는 `data/archive.json`).
- 한계: 암호화 파일은 누구나 내려받아 오프라인으로 암호를 대입할 수 있습니다. 무작위 단어 4개 이상(또는 16자 이상) 암호를 쓰세요.
- 중요: Git 기록에 이전 평문이 남아 있습니다. 저장소가 public이면 기록(history)을 지우거나 저장소를 새로 만드세요.
- HTTPS(GitHub Pages)에서만 복호화가 동작합니다(file:// 로컬 열기는 브라우저에 따라 실패).

## v6 변경
- **한글 서술**: 프롬프트를 다시 썼습니다(번역투 금지 목록, 용어 표기표, 고쳐 쓰는 예시, 분야별 4문단 14-18문장, 연구 제안에 `연구 설계`·`성공 기준` 추가). 초안 뒤에 교정 호출을 한 번 더 합니다(`llm.polish`). 교정본이 구조·인용 번호·분량을 지키지 못하면 초안을 씁니다. `llm.tone: polite`면 ~합니다 체입니다.
- **중복 제외**: 이전 호에 실린 논문(DOI, arXiv 번호, 제목이 같은 것)은 다음 호에서 뺍니다. 기록은 `data/seen.json`(공개되지 않는 폴더)입니다. 같은 날짜 재실행은 영향이 없습니다. 최초 1회는 이미 발행된 암호화 txt 문서에서 기록을 복원합니다. 끄려면 `dedupe_history: false`.
- **암호 교체**: Secret `OLD_PAGE_PASSWORD`에 직전 암호를 넣고 `PAGE_PASSWORD`를 새 암호로 바꾼 뒤 실행하면 `docs/`의 기존 페이지가 모두 새 암호로 다시 암호화됩니다. 성공하면 `data/pwcheck.json`에 지문이 저장되어 다음 주부터는 점검을 건너뜁니다. 성공 후 `OLD_PAGE_PASSWORD`는 지우세요.
- **양자 분야**: `Quantum Algorithms & NISQ`(신규)와 `Quantum Computing for Practical Applications`(응용으로 재정의) 두 분야로 나누고, 둘 다 arXiv `quant-ph`를 함께 검색합니다(`preprint.extra`).

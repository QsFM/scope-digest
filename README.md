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

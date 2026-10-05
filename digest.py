import os, re, sys, math, datetime as dt, html, itertools
import xml.etree.ElementTree as ET
from collections import Counter
import requests, yaml, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), encoding="utf-8"))
MAIL = os.environ.get("OPENALEX_MAILTO", "")
OA = "https://api.openalex.org/works"
today = dt.date.today()

def norm_text(s):
    s = (s or "").lower()
    for k, v in CFG.get("synonyms", {}).items():
        s = re.sub(r"\b%s\b" % re.escape(k.lower()), v, s)
    return s

def inv_abstract(ii):
    if not ii: return ""
    pos = {p: w for w, ps in ii.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))

def get(url, params=None, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.ok: return r
        except requests.RequestException: pass
    return None

def openalex(query, start, end, per_page=40):
    p = {"search": query, "per-page": per_page, "mailto": MAIL,
         "filter": f"from_publication_date:{start},to_publication_date:{end},type:article"}
    r = get(OA, p)
    out = []
    for w in (r.json().get("results", []) if r else []):
        out.append(dict(title=w.get("title") or "", abstract=inv_abstract(w.get("abstract_inverted_index")),
            doi=(w.get("doi") or "").replace("https://doi.org/", ""), url=w.get("doi") or w.get("id"),
            venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            date=w.get("publication_date"), preprint=False))
    return out

def oa_count(query, start, end):
    r = get(OA, {"search": query, "per-page": 1, "mailto": MAIL,
                 "filter": f"from_publication_date:{start},to_publication_date:{end}"})
    return r.json()["meta"]["count"] if r else 0

def arxiv(days):
    pc = CFG["preprint"]; since = today - dt.timedelta(days=days)
    kw = " OR ".join(f'abs:"{k}"' for k in pc["ml_only_keywords"])
    cats = " OR ".join(f"cat:{c}" for c in pc["ml_only_categories"])
    r = get("http://export.arxiv.org/api/query", {"search_query": f"({kw}) AND ({cats})",
        "sortBy": "submittedDate", "sortOrder": "descending", "max_results": 40})
    out = []
    if not r: return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(r.content).findall("a:entry", ns):
        d = e.find("a:published", ns).text[:10]
        if d < str(since): continue
        link = e.find("a:id", ns).text
        out.append(dict(title=" ".join(e.find("a:title", ns).text.split()),
            abstract=" ".join(e.find("a:summary", ns).text.split()), doi="", url=link,
            venue="arXiv (preprint, not peer-reviewed)", date=d, preprint=True))
    return out

def dedupe(ps):
    seen, out = set(), []
    for p in ps:
        if re.search(r"zenodo|figshare|dryad|osf|ssrn|researchgate", p["venue"].lower()) or p["title"].lower().startswith("data for the research"): continue
        key = re.sub(r"\W+", "", p["title"].lower())
        if key and key not in seen and p["title"]:
            seen.add(key); out.append(p)
    return out

def collect(start, end, preprints=True):
    ps = []
    for t in CFG["topics"].values():
        for k in t["keywords"]:
            ps += openalex(k, start, end)
    if preprints and CFG["preprint"]["enabled"]:
        ps += arxiv((end - start).days if isinstance(start, dt.date) else CFG["window_days"])
    return dedupe(ps)

def terms(texts, n=(1, 2)):
    v = TfidfVectorizer(stop_words="english", ngram_range=n, min_df=2, token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}")
    return v

def classify(papers):
    names = list(CFG["topics"]); prof = [norm_text(" ".join(CFG["topics"][n]["keywords"])) for n in names]
    docs = [norm_text(p["title"] + ". " + p["title"] + ". " + p["abstract"]) for p in papers]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = v.fit_transform(prof + docs); T, D = X[:len(names)], X[len(names):]
    S = cosine_similarity(D, T)
    for i, p in enumerate(papers):
        p["topic"] = names[int(S[i].argmax())]; p["rel"] = float(S[i].max())
    return [p for p in papers if p["rel"] >= CFG["min_relevance"]]

def top_terms(docs, k=8):
    if len(docs) < 2: return []
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=3000)
    X = v.fit_transform(docs); s = np.asarray(X.sum(0)).ravel(); f = v.get_feature_names_out()
    return [f[i] for i in s.argsort()[::-1][:k]]

def rising(cur, base, k=10):
    def df(ps):
        c = Counter()
        for p in ps:
            ws = set(re.findall(r"[a-z][a-z\-]{3,}(?: [a-z][a-z\-]{3,})?", norm_text(p["title"] + " " + p["abstract"])))
            c.update(w for w in ws if not set(w.split()) & ENG_STOP)
        return c
    c, b = df(cur), df(base); n1, n0 = max(len(cur), 1), max(len(base), 1)
    sc = []
    for w, x in c.items():
        if x < 3: continue
        p0 = (b.get(w, 0) + 1) / (n0 + 2); p1 = x / n1
        se = math.sqrt(p0 * (1 - p0) / n1 + 1e-9)
        sc.append(((p1 - p0) / se, w, x, b.get(w, 0)))
    return sorted(sc, reverse=True)[:k]

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as ENG_STOP
ENG_STOP = set(ENG_STOP) | {"using", "based", "paper", "study", "results", "show", "proposed", "method", "approach", "also", "new", "which", "these", "such", "that", "this", "with"}

def gaps(k=8):
    """내 연구 키워드 × 이웃 개념 교차 갭: 관측 공출현 / 기대 공출현이 낮은 쌍."""
    s, e = today - dt.timedelta(days=365), today
    mine = ["bayesian optimization", "quantum annealing", "factorization machine", "metasurface", "active learning", "sim-to-real"]
    ext = ["gate-based quantum computing", "large language model", "diffusion model", "graph neural network",
           "physics-informed neural network", "reinforcement learning", "transformer", "foundation model", "uncertainty quantification"]
    cache = {}
    def c(q):
        if q not in cache: cache[q] = oa_count(q, s, e)
        return cache[q]
    N = max(oa_count("optimization", s, e), 1) * 3
    res = []
    for a, b in itertools.product(mine, ext):
        ca, cb = c(a), c(b)
        if min(ca, cb) < 50: continue
        ab = c(f'"{a}" "{b}"'); exp = ca * cb / N
        res.append((ab / (exp + 1e-9), a, b, ca, cb, ab, exp))
    return sorted(res)[:k]

def evidence(a, b, papers):
    best = None
    for p in papers:
        t = norm_text(p["title"] + " " + p["abstract"])
        sc = sum(w in t for w in a.split()) + sum(w in t for w in b.split())
        if sc and (not best or sc > best[0]): best = (sc, p)
    return best[1] if best else None


# ---------- 선택형 LLM 단계 (GEMINI_API_KEY 가 있을 때만) ----------
def llm_section(papers):
    key = os.environ.get("GEMINI_API_KEY")
    L = CFG.get("llm", {})
    if not key or not L.get("enabled", True) or not papers: return None
    import json
    sel = []
    for tn in CFG["topics"]:
        ps = sorted([p for p in papers if p["topic"] == tn], key=lambda p: -p["rel"])[:L.get("papers_per_topic", 6)]
        sel += ps
    ids = {f"P{i+1}": p for i, p in enumerate(sel)}
    lines = [f"[{k}] ({p['topic']}) {p['title']} :: {' '.join(re.split(r'(?<=[.!?]) ', p['abstract'])[:3])[:450]}" for k, p in ids.items()]
    lab = "; ".join(f"{n}: {', '.join(v['keywords'][:5])}" for n, v in CFG["topics"].items())
    prompt = ("You are a research analyst for an electrical-engineering lab. Research areas: " + lab + ".\n"
        "Below are recent papers. Use ONLY these papers. Cite them by id like P3. Do not invent papers or facts.\n"
        "Return JSON: {\"trends\":[{\"topic\":str,\"summary\":str(Korean, 3-4 sentences),\"cited\":[ids]}],"
        "\"proposals\":[{\"title\":str(Korean),\"rationale\":str(Korean, 2-3 sentences, link to the lab's areas),\"novelty\":1-5,\"cited\":[ids]}]}\n"
        "Give 3-5 proposals that combine at least two lab areas or transfer a method from one paper to a lab area.\n\n" + "\n".join(lines))
    for model in L.get("models", ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-2.5-flash"]):
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"}, timeout=120,
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}})
            if not r.ok:
                print("llm", model, r.status_code); continue
            data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            return model, data, ids
        except Exception as e:
            print("llm fail", model, type(e).__name__)
    return None

def render_llm(res):
    if not res: return ""
    model, data, ids = res
    def cites(c):
        ok = [x for x in c if x in ids]          # 수집 목록에 없는 id는 제거
        return " ".join(f"<a href='{ids[x]['url']}' title='{html.escape(ids[x]['title'])}'>[{x}]</a>" for x in ok)
    h = [f"<h3>AI 동향 요약 <small>(생성: {model}, 수집 논문만 인용, 타당성은 직접 확인)</small></h3>"]
    for t in data.get("trends", []):
        h.append(f"<p><b>{html.escape(str(t.get('topic','')))}</b>: {html.escape(str(t.get('summary','')))} {cites(t.get('cited', []))}</p>")
    h.append("<h3>AI 미래 연구 제안</h3><ol>")
    for p in data.get("proposals", []):
        h.append(f"<li><b>{html.escape(str(p.get('title','')))}</b> (새로움 {html.escape(str(p.get('novelty','?')))}/5)<br>{html.escape(str(p.get('rationale','')))} {cites(p.get('cited', []))}</li>")
    h.append("</ol>")
    return "".join(h)

def render(papers, base, gp, extra=''):
    h = [f"<h2>SCOPE Weekly Digest — {today}</h2><p><a href='archive.html'>지난 호 보기</a></p>",
         f"<p style='color:#666'>수집 {len(papers)}편 (관련도 필터 후). 요약은 알고리즘 기반(초록 키워드)이며 LLM을 사용하지 않았습니다. 프리프린트는 [PRE]로 표시.</p>"]
    h.append(extra)
    for tn in CFG["topics"]:
        ps = sorted([p for p in papers if p["topic"] == tn], key=lambda p: -p["rel"])
        if not ps: continue
        tt = ", ".join(top_terms([norm_text(p["title"] + " " + p["abstract"]) for p in ps]))
        h.append(f"<h3>{html.escape(tn)} <small>({len(ps)}편)</small></h3><p><b>이번 기간 핵심어:</b> {html.escape(tt)}</p><ul>")
        for p in ps[:CFG["top_n_per_topic"]]:
            tag = "[PRE] " if p["preprint"] else ""
            ab = re.split(r"(?<=[.!?]) ", p["abstract"]); gist = " ".join(ab[-2:])[:300]
            h.append(f"<li>{tag}<a href='{p['url']}'>{html.escape(p['title'])}</a><br><small>{html.escape(p['venue'])}, {p['date']}</small><br><small>{html.escape(gist)}</small></li>")
        h.append("</ul>")
    h.append("<h3>급상승 키워드 (직전 기간 대비 z-score)</h3><ol>")
    for z, w, x, b in rising(papers, base):
        h.append(f"<li>{html.escape(w)} — 이번 {x}편 / 이전 {b}편 (z={z:.1f})</li>")
    h.append("</ol><h3>미래 연구 후보 (교차 갭: 관측 공출현 / 기대 공출현이 낮은 쌍)</h3><ol>")
    for r, a, b, ca, cb, ab, exp in gp:
        ev = evidence(a, b, papers)
        evs = f"<br><small>근거 논문: <a href='{ev['url']}'>{html.escape(ev['title'])}</a></small>" if ev else ""
        h.append(f"<li><b>{html.escape(a)} × {html.escape(b)}</b> — 최근 1년 {ca}편, {cb}편, 교차 {ab}편 (기대 {exp:.1f}편){evs}</li>")
    h.append("</ol><p style='color:#999'>후보는 키워드 조합이며, 타당성 판단은 직접 하셔야 합니다.</p>")
    return "<html><body style='font-family:sans-serif;max-width:760px'>" + "".join(h) + "</body></html>"

def publish(body):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"); os.makedirs(d, exist_ok=True)
    name = f"{today}"
    open(f"{d}/{name}.html", "w", encoding="utf-8").write(body)
    txt = re.sub(r"<li>", "\n- ", body); txt = re.sub(r"<h[23]>", "\n\n## ", txt)
    txt = re.sub(r"<br>|</p>|</ul>|</ol>", "\n", txt); txt = html.unescape(re.sub(r"<[^>]+>", "", txt))
    open(f"{d}/{name}.txt", "w", encoding="utf-8").write(txt.strip() + "\n")
    open(f"{d}/index.html", "w", encoding="utf-8").write(body)  # 최신호
    items = sorted({f[:10] for f in os.listdir(d) if re.match(r"\d{4}-\d\d-\d\d\.", f)}, reverse=True)
    arc = "".join(f"<li>{x} — <a href='{x}.html'>HTML</a> · <a href='{x}.txt'>TXT</a></li>" for x in items)
    open(f"{d}/archive.html", "w", encoding="utf-8").write(
        "<html><body style='font-family:sans-serif'><h2>SCOPE Digest 아카이브</h2><p><a href='index.html'>최신호</a></p><ul>" + arc + "</ul></body></html>")

if __name__ == "__main__":
    w, b = CFG["window_days"], CFG["baseline_days"]
    cur = classify(collect(today - dt.timedelta(days=w), today))
    base = collect(today - dt.timedelta(days=w + b), today - dt.timedelta(days=w), preprints=False)
    body = render(cur, base, gaps(), render_llm(llm_section(cur)))
    open("digest.html", "w", encoding="utf-8").write(body)
    if "--dry-run" in sys.argv: print("digest.html written,", len(cur), "papers")
    else: publish(body); print("published")

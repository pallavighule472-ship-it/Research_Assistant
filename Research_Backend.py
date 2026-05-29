import os
import re
import time
import operator
import concurrent.futures
from typing import TypedDict, List, Annotated
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
import wikipedia as _wiki_pkg
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "Research-Assistant"

try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# Domains blocked from crawling AND from appearing in references
SKIP_DOMAINS = {
    "w3.org", "schema.org", "tavily.com",
    "youtube.com", "youtu.be",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "pinterest.com",
    "blogspot.com", "wordpress.com",
    # Generic/non-informational domains that pollute search results
    "dictionary.cambridge.org", "merriam-webster.com",
    "global.com", "globalplayer.com", "globle-game.com",
    "amazon.com", "ebay.com", "etsy.com",
}

# LLM models
planner_model  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
extract_model  = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1500)
evaluate_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2500)

# Tavily search tool (optional — requires TAVILY_API_KEY)
tavily_search = TavilySearch(max_results=5) if (TavilySearch and os.getenv("TAVILY_API_KEY")) else None


def crawl_webpage(url: str) -> str:
    """Fetches a webpage and returns clean extracted text, stripping HTML boilerplate."""
    if not url.startswith("http"):
        return ""
    if url.endswith((".pdf", ".zip", ".png", ".jpg", ".jpeg", ".mp4")):
        return ""
    parsed_netloc = urlparse(url).netloc.lstrip("www.")
    if any(d in parsed_netloc for d in SKIP_DOMAINS):
        return ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        html = response.text
        if _BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "form", "header", "aside"]):
                element.decompose()
            text = soup.get_text(separator="\n")
        else:
            text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '\n', text)
        lines = [line.strip() for line in text.splitlines()]
        chunks = [line for line in lines if len(line) > 20]
        clean_text = "\n".join(chunks)[:20000]
        print(f"  [+] Crawled: {url} ({len(clean_text)} chars extracted)")
        return clean_text
    except Exception as e:
        print(f"  [!] Failed to crawl {url}: {e}")
        return ""


class ResearchState(TypedDict):
    user_query: str
    target_language: str
    writing_style: str
    target_length: str
    plan: list
    notes: list
    iteration: int
    max_iteration: int
    web_results: str
    web_links: Annotated[list, operator.add]
    final_answer: str
    is_complete: bool


class ResearchPlan(BaseModel):
    steps: List[str] = Field(description="A list of specific search queries or research steps needed to answer the user's query.")


class ResearchEvaluation(BaseModel):
    is_complete: bool = Field(description="True if the research fully answers the user's query, False if more research is needed.")
    reasoning: str = Field(description="Explanation of why the research is complete or what is missing.")


def user_query_node(state: ResearchState) -> ResearchState:
    query = state["user_query"]
    print(f"--- Analyzing User Query: '{query}' ---")
    structured_llm = planner_model.with_structured_output(ResearchPlan)
    messages = [
        SystemMessage(content=f"""You are an expert research strategist. Your sole objective is to produce a set of topics/sections that together give a complete, well-structured answer to this specific question:

"{query}"

Generate 5 to 7 section topics. Each topic will serve two purposes: (1) a web search query to gather information, and (2) a section title in the final response. So every topic must be both searchable AND meaningful as a standalone section.

STRUCTURE RULES:
- If the query is broad or general (e.g. "inflammation in body", "how does X work", "what is Y"), ALWAYS start with foundational sections:
  • First: definition, types, and how it works (the basics — never skip this)
  • Second: causes and triggers
  • Third: effects, symptoms, or real-world impact
  • Then: diagnosis, treatment, or management
  • Then: advanced topics, debates, or recent research ONLY after basics are covered
- If the query is specific or niche, derive sections directly from what is being asked — no forced structure.
- Do NOT jump straight to debates, case studies, or advanced research if the user is asking a general question. Cover the fundamentals first.
- Do NOT anchor topics to a specific year (e.g. "in 2023") unless the user explicitly asked.
- Output only the list — no explanations."""),
        HumanMessage(content=f"Question to answer: {query}")
    ]
    default_plan = [f"{step}: {query}" for step in _DEFAULT_PLAN]
    try:
        plan_output = structured_llm.invoke(messages)
        plan = plan_output.steps
        print(f"Generated research plan: {plan}")
    except Exception as e:
        print(f"Error generating plan: {e}. Using default plan.")
        plan = default_plan
    if not plan or (len(plan) == 1 and plan[0].strip().lower() == query.strip().lower()):
        print("  [!] Degenerate plan detected. Using default plan.")
        plan = default_plan
    return {
        "iteration": 0,
        "web_results": "",
        "final_answer": "",
        "plan": plan,
        "notes": [],
        "is_complete": False,
    }


def web_search(state: ResearchState) -> ResearchState:
    """Searches DuckDuckGo, Wikipedia, and Tavily in parallel then crawls discovered URLs."""
    plan = state.get("plan", [])
    query = state.get("user_query", "")
    search_queries = plan if plan else [query]
    print(f"--- Executing web search for {len(search_queries)} queries ---")

    ddg_snippets = []
    pre_crawled = []
    discovered_links = []
    pre_crawled_urls = set()
    ddg_hrefs = []

    def perform_single_search(q, use_wikipedia=False):
        print(f"  Searching: {q}")

        def _run_tavily():
            blocks, urls, snips = [], [], []
            if not tavily_search:
                return blocks, urls, snips
            try:
                t_results = tavily_search.invoke(q)
                if isinstance(t_results, list):
                    for item in t_results:
                        url = item.get("url", "")
                        content = item.get("content", "")
                        if content and url:
                            blocks.append(f"### Web Source: {url}\n{content}")
                            urls.append(url)
                    print(f"  [+] Tavily: {len(blocks)} results")
                elif t_results:
                    snips.append(str(t_results))
            except Exception as e:
                print(f"  [!] Tavily failed for '{q}': {e}")
            return blocks, urls, snips

        def _run_ddg():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=5))
                snippet = " ".join(r.get("body", "") for r in results)
                hrefs = [r["href"] for r in results if r.get("href") and not any(d in r["href"] for d in SKIP_DOMAINS)]
                print(f"  [+] DuckDuckGo: {len(results)} results, {len(hrefs)} source URLs")
                return snippet, hrefs
            except Exception as e:
                print(f"  [!] DuckDuckGo failed for '{q}': {e}")
                return "", []

        def _run_wiki():
            try:
                try:
                    page = _wiki_pkg.page(q, auto_suggest=True)
                except _wiki_pkg.exceptions.DisambiguationError as e:
                    page = _wiki_pkg.page(e.options[0])
                article_url = page.url
                summary = page.summary[:2000]
                print(f"  [+] Wikipedia: OK ({page.title})")
                return f"### Web Source: {article_url}\n{summary}", article_url
            except Exception as e:
                print(f"  [!] Wikipedia failed for '{q}': {e}")
                return "", ""

        n_inner = 3 if use_wikipedia else 2
        snippets, crawled_blocks, crawled_urls, ddg_source_urls = [], [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_inner) as inner:
            f_tavily = inner.submit(_run_tavily)
            f_ddg    = inner.submit(_run_ddg)
            f_wiki   = inner.submit(_run_wiki) if use_wikipedia else None
            try:
                t_blocks, t_urls, t_snips = f_tavily.result(timeout=20)
                crawled_blocks.extend(t_blocks)
                crawled_urls.extend(t_urls)
                snippets.extend(t_snips)
            except Exception:
                pass
            try:
                ddg_snip, ddg_hrefs = f_ddg.result(timeout=20)
                if ddg_snip:
                    snippets.append(ddg_snip)
                ddg_source_urls.extend(ddg_hrefs)
            except Exception:
                pass
            if f_wiki:
                try:
                    w_block, w_url = f_wiki.result(timeout=20)
                    if w_block:
                        crawled_blocks.append(w_block)
                        crawled_urls.append(w_url)
                except Exception:
                    pass
        return snippets, crawled_blocks, crawled_urls, ddg_source_urls

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, max(1, len(search_queries)))) as executor:
        futures = {
            executor.submit(perform_single_search, q, i == 0): q
            for i, q in enumerate(search_queries)
        }
        try:
            for future in concurrent.futures.as_completed(futures, timeout=60):
                q = futures[future]
                try:
                    snippets, crawled_blocks, crawled_urls, batch_hrefs = future.result()
                    ddg_snippets.extend(snippets)
                    pre_crawled.extend(crawled_blocks)
                    discovered_links.extend(crawled_urls)
                    pre_crawled_urls.update(crawled_urls)
                    ddg_hrefs.extend(batch_hrefs)
                except Exception as exc:
                    print(f"  [!] Search thread failed for '{q}': {exc}")
        except concurrent.futures.TimeoutError:
            print("  [!] Search executor wall-time exceeded 60s — proceeding with results gathered so far.")

    combined_snippets = "\n\n".join(ddg_snippets)
    unique_urls = []
    for url in ddg_hrefs:
        if (url not in unique_urls
                and url not in pre_crawled_urls
                and not any(domain in url for domain in SKIP_DOMAINS)):
            unique_urls.append(url)

    top_urls = unique_urls[:6]
    print(f"[+] Tavily/Wikipedia provided {len(pre_crawled)} blocks. Crawling {len(top_urls)} new DDG URLs...")

    if top_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(top_urls))) as executor:
            future_to_url = {executor.submit(crawl_webpage, url): url for url in top_urls}
            try:
                for future in concurrent.futures.as_completed(future_to_url, timeout=45):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            pre_crawled.append(f"### Web Source: {url}\n{content}")
                            # Only register as a citation source if crawl returned useful content
                            if url not in discovered_links:
                                discovered_links.append(url)
                    except Exception as exc:
                        print(f"  [!] Crawler thread error for {url}: {exc}")
            except concurrent.futures.TimeoutError:
                print("  [!] Crawler wall-time exceeded 45s — skipping remaining URLs.")

    if not pre_crawled:
        print("[!] All sources returned no content. Using raw snippets.")
        final_web_results = combined_snippets
    else:
        final_web_results = "\n\n=========================================\n\n".join(pre_crawled)

    return {
        "web_results": final_web_results,
        "web_links": discovered_links,
    }


def extract_info(state: ResearchState) -> ResearchState:
    """Extracts structured facts from each web source block in parallel."""
    query = state["user_query"]
    web_results = state["web_results"]
    print("--- Extracting Information from Web Results ---")
    sources = web_results.split("=========================================\n\n")
    extracted_notes_list = []

    system_content = """You are a professional academic research retrieval agent.
Your job is to extract the highest-quality, most substantive, and factually precise information from raw source text.

INSTRUCTIONS:
1. Focus only on relevant, factual, and recent information related to the query.
2. Extract key facts, statistics, named studies, specific figures, definitions, mechanisms, and direct claims — preserve specificity.
3. Prioritize: quantitative data (percentages, counts, dates, measurements), named researchers or institutions, causal mechanisms, methodological details, and study outcomes.
4. Ignore navigation menus, site footers, cookie notices, ads, promotional language, or irrelevant boilerplate.
5. Deduplicate information and structure it clearly with bullet points.
6. Do NOT write a final report yet. Just list raw facts cleanly.
7. If the source text contains no relevant information, output nothing or "No relevant information".
8. NEVER hallucinate or add facts not present in the text — extract only what is explicitly stated.
9. If two sources contradict each other on a fact, explicitly flag it as: [CONFLICT: Source A says X, Source B says Y]
10. Note the publication date or recency of information whenever visible in the source text.
11. For technical or scientific sources, preserve precise terminology, model names, compound names, or methodology labels — do not simplify them away."""

    def extract_from_source(source_index, source_text):
        source_text = source_text.strip()
        if not source_text or len(source_text) < 200:
            return None
        source_url = ""
        lines = source_text.split("\n", 2)
        if lines[0].startswith("### Web Source:"):
            source_url = lines[0].replace("### Web Source:", "").strip()
            source_text = "\n".join(lines[1:]) if len(lines) > 1 else source_text
        print(f"  [-] Extracting facts from source {source_index + 1}/{len(sources)} ({source_url or 'unknown'})...")
        sample_text = source_text[:10000]
        human_content = f"USER QUERY:\n{query}\n\nSOURCE URL: {source_url}\n\nSOURCE RAW TEXT:\n{sample_text}"
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        try:
            response = extract_model.invoke(messages)
            content = response.content.strip()
            if content and "No relevant information" not in content:
                tag = f" (source: {source_url})" if source_url else ""
                return f"### Facts from Source{tag}:\n{content}"
        except Exception as e:
            print(f"  [!] Extraction failed for source {source_index + 1}: {e}")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as executor:
        futures = [executor.submit(extract_from_source, i, s) for i, s in enumerate(sources)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                extracted_notes_list.append(res)

    new_notes = extracted_notes_list if extracted_notes_list else ["Failed to extract unique facts from crawled sources."]
    return {
        "notes": state["notes"] + new_notes,
        "web_results": "",
    }


_DEFAULT_PLAN = [
    "Overview, definition, and key concepts",
    "Background, history, and context",
    "Key aspects, components, and notable examples",
    "Practical implications, applications, and use cases",
    "Challenges, limitations, and open questions",
]


def _invoke_with_retry(llm, msgs: list, max_attempts: int, label: str) -> str:
    """Calls llm.invoke(msgs) with exponential backoff. Returns empty string if all attempts fail."""
    for attempt in range(max_attempts):
        try:
            return llm.invoke(msgs).content.strip()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "rate limit" in err_str.lower() or "429" in err_str or "quota" in err_str.lower()
            wait = 30 if is_rate_limit else (2 ** attempt * 4)
            print(f"  [!] {label} attempt {attempt + 1}/{max_attempts} failed: {err_str[:120]}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""


_STYLE_MAP = {
    "Book-Style Detailed": "Write in a richly narrative, book-chapter style — eloquent flowing prose, deep storytelling, and immersive detail.",
    "Academic Thesis":     "Write in a formal academic register — precise argumentation, heavy citations, structured methodology, and rigorous analysis.",
    "Executive Summary":   "Write in a concise business-professional tone — key findings first, minimal jargon, bullet-friendly structure, actionable insights.",
    "Casual & Engaging":   "Write in a clear conversational tone — accessible language, relatable analogies, and an engaging narrative anyone can follow.",
}

# (para_desc, min_words_per_section, max_tokens_per_section, meta_max_tokens)
_LENGTH_MAP = {
    "Deep Book (6+ paragraphs per section)": ("at least 6 paragraphs (120+ words each)", 600, 1600, 1100),
    "Medium-depth (3-4 paragraphs)":          ("4 to 5 paragraphs (80–120 words each)",   400, 1100,  900),
    "Concise Summary (1-2 paragraphs)":       ("2 to 3 paragraphs (60–80 words each)",    200,  600,  700),
}


def _write_report_parallel(
    plan: list,
    query: str,
    shared_notes: str,
    target_language: str,
    style_desc: str,
    para_desc: str,
    min_per_sec: int,
    sec_tok: int,
    meta_tok: int,
    callbacks=None,
) -> tuple:
    """Writes opening, all body sections, and closing concurrently. Returns (opening, section_results, closing)."""
    _cb = callbacks or []
    section_writer = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=sec_tok, streaming=True, callbacks=_cb)
    meta_writer    = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=meta_tok, streaming=True, callbacks=_cb)

    def write_opening():
        msgs = [
            SystemMessage(content=f"""You are writing the opening section for a comprehensive response to this query: "{query}"

Read the query carefully and choose the opening format that genuinely serves it best:

- If it is an academic or research question (e.g. "What causes inflation?", "Explain CRISPR", "History of AI") →
  Write a formal Abstract following IMRAD structure (Background, Objective, Methods, Key Findings, Implications), minimum 200 words, plus **Keywords**: 6–8 terms. Be dense — name specific findings and data, no vague filler.

- If it is a practical, list-based, or guide question (e.g. "Best places to visit in Japan", "How to learn Python", "Top budget laptops") →
  Write a short, friendly introduction (2–3 paragraphs) that orients the reader: what they will find, why it matters, and a quick overview. Sound like a knowledgeable guide, not an academic.

- If it is a how-to or step-by-step question →
  Write a brief overview covering what the process achieves, what prerequisites are needed, and what the reader will know by the end.

- If it is a comparison or "X vs Y" question →
  Write a framing paragraph that sets the scope of the comparison, what criteria matter, and what conclusion the reader can expect.

Use your judgment — the format must match what this specific query is asking. Do NOT apply a one-size-fits-all template.
CITATION ACCURACY: Never state a specific publication year for a study unless that year is explicitly present in the research notes.
Language: {target_language}. Style: {style_desc}. Write only the opening section."""),
            HumanMessage(content=f"QUERY: {query}\n\n{shared_notes}"),
        ]
        return _invoke_with_retry(meta_writer, msgs, 3, "Opening")

    def write_section(idx, section_title):
        other_topics = [s for i, s in enumerate(plan) if i != idx]
        avoid_note   = f"Other sections cover: {', '.join(other_topics)}. Do NOT repeat or overlap their content — focus exclusively on '{section_title}'."
        msgs = [
            SystemMessage(content=f"""You are writing section {idx + 1} of {len(plan)} for a comprehensive response to: "{query}"

Section topic: "{section_title}"

ADAPTIVE FORMAT RULE — choose the format that best serves this specific query and section:
- Recommendations / places / products / options → Use **Bold Name** for each item, followed by 3–5 concrete bullet points with specific details (location, cost, timings, insider tips, what to expect). No vague descriptions.
- Explanations / mechanisms / science / history → Use flowing analytical paragraphs. Cite specific facts, studies, statistics, and named sources with **[n]**. Explain causes, mechanisms, and scholarly debate — not just surface summaries.
- Step-by-step processes / how-tos → Use numbered steps with clear, actionable instructions and explanations for each step.
- Comparisons → Use structured analysis with clear criteria; highlight key differences with evidence.
- Mixed queries → Combine formats as needed — bold + bullets for enumerable items, paragraphs for context and analysis.

QUALITY REQUIREMENTS (apply regardless of format):
- Every item or claim must be specific and concrete — no vague filler.
- For factual/research content: cite statistics, named studies, or specific data with **[n]** from the numbered sources list.
- For practical content: include real details — names, numbers, locations, prices, timelines, tips.
- Minimum {min_per_sec} words for this section.
- {style_desc}
- Language: {target_language}
- FOCUS: Every sentence must directly help answer "{query}". Cut anything that doesn't serve the question.
- {avoid_note}
- INTEGRITY RULE: Do NOT invent fictional case studies, hypothetical patients, or fabricated examples. Only describe studies or cases that are explicitly present in the research notes. If none exist, discuss real named research findings instead.
- CITATION ACCURACY: Never state a specific publication year (e.g. "a 2023 study") unless that year is explicitly mentioned in the research notes for that source. Writing an incorrect year is worse than omitting it.
- Start directly with: ## {section_title}"""),
            HumanMessage(content=f"QUERY: {query}\n\n{shared_notes}\n\nREMINDER: minimum {min_per_sec} words. Choose the right format for this query — be specific, be useful."),
        ]
        content = _invoke_with_retry(section_writer, msgs, 4, f"Section {idx + 1}/{len(plan)}")
        if content:
            print(f"  [+] Section {idx + 1}/{len(plan)} done: {section_title[:50]}")
            return idx, content
        print(f"  [!!] Section '{section_title[:50]}' failed — check API key/quota.")
        return idx, f"## {section_title}\n\n*Section unavailable.*"

    def write_closing():
        msgs = [
            SystemMessage(content=f"""You are writing the closing section for a comprehensive response to: "{query}"

Read the query and choose the closing format that best serves it:

- Academic / research / analysis question →
  Write: (1) ## Limitations — specific methodological gaps, data constraints, scope limits (150+ words, precise — not vague disclaimers). (2) ## Conclusion — synthesize findings and their significance, what they mean together, implications for theory or practice (300+ words). (3) ## Future Directions — 4–6 specific open questions or research avenues.

- Practical guide / list / travel / recommendation question →
  Write: (1) ## Quick Tips — 6–8 concise, actionable bullets the reader should know. (2) ## Summary — 2–3 paragraph wrap-up of the key takeaways and a clear recommendation or final thought.

- How-to / step-by-step question →
  Write: (1) ## Common Mistakes to Avoid — 4–6 bullets on what goes wrong and how to prevent it. (2) ## Summary — what the reader should now be able to do and suggested next steps.

- Comparison / X vs Y question →
  Write: (1) ## Verdict — a clear, evidence-based recommendation of which is better and for whom, with reasoning. (2) ## Summary — brief recap of the key decision factors.

Use your judgment — write the closing that is genuinely most useful for this query.
Language: {target_language}. Style: {style_desc}. Do NOT repeat section content verbatim — synthesize and elevate."""),
            HumanMessage(content=f"QUERY: {query}\n\n{shared_notes}"),
        ]
        result = _invoke_with_retry(meta_writer, msgs, 3, "Closing")
        return result or "## Summary\n\n*Summary unavailable.*"

    print(f"--- Writing opening + {len(plan)} sections + closing in parallel ---")
    section_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_opening  = executor.submit(write_opening)
        f_closing  = executor.submit(write_closing)
        f_sections = []
        for i, s in enumerate(plan):
            if i > 0:
                time.sleep(1)
            f_sections.append(executor.submit(write_section, i, s))

    opening_text = f_opening.result()
    closing_text = f_closing.result()
    for f in f_sections:
        idx, content = f.result()
        section_results[idx] = content

    return opening_text, section_results, closing_text


def _append_references(draft: str, unique_links: list) -> str:
    """Replaces any LLM-generated References section with our numbered format."""
    quality_links = [l for l in unique_links if not any(d in l for d in SKIP_DOMAINS)]
    if not quality_links:
        return draft
    # Strip any References section the LLM may have written (format varies; ours is canonical)
    draft = re.sub(r'\n*---\n*##\s+References\b.*$', '', draft, flags=re.DOTALL | re.IGNORECASE)
    draft = re.sub(r'\n*##\s+References\b.*$', '', draft, flags=re.DOTALL | re.IGNORECASE)
    draft += "\n\n---\n\n## References\n\n"
    for i, link in enumerate(quality_links, 1):
        domain = urlparse(link).netloc or link
        draft += f"[{i}] *{domain}*. Available at: {link}\n\n"
    return draft


def research_writer(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Orchestrates parallel section writing and reference assembly into the final report."""
    query  = state["user_query"]
    notes  = state.get("notes", [])
    links  = state.get("web_links", [])
    plan   = state.get("plan", [])
    target_language = state.get("target_language", "English")
    writing_style   = state.get("writing_style", "Book-Style Detailed")
    target_length   = state.get("target_length", "Medium-depth (3-4 paragraphs)")

    print(f"--- Research Writer: parallel sections | {writing_style} | {target_length} ---")

    style_desc = _STYLE_MAP.get(writing_style, _STYLE_MAP["Book-Style Detailed"])
    para_desc, min_per_sec, sec_tok, meta_tok = _LENGTH_MAP.get(
        target_length, _LENGTH_MAP["Medium-depth (3-4 paragraphs)"]
    )

    compiled_notes = "\n\n".join(n for n in notes if n and n.strip())
    compiled_notes = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', compiled_notes)
    if len(compiled_notes) > 28000:
        compiled_notes = compiled_notes[:28000] + "\n\n[...notes truncated for length...]"

    unique_links = list(dict.fromkeys(links))
    numbered_sources = ""
    if unique_links:
        numbered_sources = "\n\nNUMBERED SOURCES LIST (cite inline as [1], [2], [3] etc.):\n"
        for i, link in enumerate(unique_links, 1):
            numbered_sources += f"[{i}] {link}\n"
    shared_notes = f"RESEARCH NOTES:\n{compiled_notes}{numbered_sources}"

    callbacks = (config.get("callbacks") or []) if config else []
    opening, section_results, closing = _write_report_parallel(
        plan, query, shared_notes, target_language,
        style_desc, para_desc, min_per_sec, sec_tok, meta_tok,
        callbacks=callbacks,
    )

    sections_body = "\n\n".join(section_results[i] for i in sorted(section_results))
    parts = [f"# {query}"]
    if opening:
        parts.append(opening)
    parts.append(sections_body)
    parts.append(closing)
    final_draft = _append_references("\n\n".join(parts), unique_links)

    print("--- Research report assembled successfully ---")
    return {"final_answer": final_draft}


def summarize_notes(state: ResearchState) -> ResearchState:
    """Compresses accumulated note blocks into a single dense knowledge base."""
    notes = state.get("notes", [])
    if not notes or len(notes) <= 1:
        print("--- Summarize Notes: Not enough note blocks to compress yet ---")
        return {}
    print(f"--- Compressing {len(notes)} blocks of notes into a unified knowledge base ---")
    compiled_notes = "\n\n--- NOTE BLOCK ---\n\n".join(n for n in notes if n and n.strip())
    if len(compiled_notes) > 32000:
        compiled_notes = compiled_notes[:32000] + "\n\n[...notes truncated for length...]"
    user_query = state.get("user_query", "")
    system_content = f"""You are an expert knowledge manager. Your goal is to compress and organize a collection of raw research notes into a single, highly dense, and deduplicated knowledge base that directly answers this question:

"{user_query}"

INSTRUCTIONS:
1. PRIORITIZE information that directly answers or supports answering the question above — facts about the exact topic, entity, event, or mechanism asked about.
2. Combine all related facts from the various note blocks.
3. Completely remove any duplicated information or redundant statements.
4. Preserve EVERY unique fact, statistic, and source reference that is relevant to the question.
5. Organize the information logically with clear bullet points.
6. Do NOT write a report or use conversational language. This is a dense factual reference for another AI.
7. Retain all context — do not summarize away critical details specific to the question, just remove fluff and overlap.
8. Preserve any [CONFLICT: ...] flags from the source notes — do not resolve them, just consolidate.
9. Where possible, keep a brief source tag like (from: domain.com) next to key statistics.
10. If a note block contains mostly off-topic background unrelated to the question, condense it heavily and keep only directly relevant facts."""
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"QUESTION TO ANSWER:\n{user_query}\n\nRAW RESEARCH NOTES:\n{compiled_notes}")
    ]
    try:
        response = evaluate_model.invoke(messages)
        print("Notes successfully summarized and compressed.")
        return {"notes": [response.content]}
    except Exception as e:
        print(f"Error summarizing notes: {e}")
        return {"notes": notes}


def evaluate(state: ResearchState) -> ResearchState:
    """Evaluates whether the accumulated research is sufficient and increments the iteration counter."""
    query = state["user_query"]
    notes = state.get("notes", [])
    iteration = state.get("iteration", 0)
    max_iteration = state.get("max_iteration", 3)
    print(f"--- Evaluating Research (Iteration {iteration + 1}/{max_iteration}) ---")
    compiled_notes = "\n\n".join(n for n in notes if n and n.strip()) if notes else "No notes gathered yet."
    structured_llm = evaluate_model.with_structured_output(ResearchEvaluation)
    messages = [
        SystemMessage(content=f"""You are a rigorous research quality evaluator. The user asked: "{query}"

Assess the research notes against these specific completeness criteria:
1. RELEVANCE: Does the research directly and specifically answer "{query}", or does it drift into generic background about the topic without addressing the actual question?
2. COVERAGE: Are all major subtopics of the query addressed?
3. DEPTH: Are key claims backed by facts, statistics, or examples?
4. RECENCY: Does it include recent information (last 1-3 years) for time-sensitive topics?
5. BALANCE: Are multiple perspectives or counterarguments represented?
6. GAPS: Are there obvious important questions left unanswered?
Only mark is_complete=True if ALL 6 criteria are satisfactorily met. RELEVANCE is the most important criterion — generic background that doesn't answer the specific question is a hard failure.
In your reasoning, explicitly state which criteria pass or fail.
"""),
        HumanMessage(content=f"USER QUERY:\n{query}\n\nCURRENT RESEARCH NOTES:\n{compiled_notes}")
    ]
    try:
        evaluation = structured_llm.invoke(messages)
        print(f"Evaluation Reasoning: {evaluation.reasoning}")
        is_complete = evaluation.is_complete
    except Exception as e:
        print(f"Error evaluating research: {e}")
        is_complete = False
    print(f"Research Complete? {is_complete}")
    return {
        "iteration": min(iteration + 1, max_iteration),
        "is_complete": is_complete,
    }


def save_notes(state: ResearchState) -> ResearchState:
    """Logs a research completion summary to console."""
    query = state.get("user_query", "")
    iterations = state.get("iteration", 0)
    sources = len(state.get("web_links", []))
    notes_size = sum(len(n) for n in state.get("notes", []))
    print(f"\n{'='*50}")
    print(f"  RESEARCH COMPLETE")
    print(f"  Query          : {query[:60]}")
    print(f"  Iterations used: {iterations}")
    print(f"  Sources crawled: {sources}")
    print(f"  Knowledge base : {notes_size:,} chars")
    print(f"{'='*50}\n")
    return {}


def optimize_research(state: ResearchState) -> ResearchState:
    """Identifies gaps in the current research and generates targeted gap-fill queries."""
    print("--- Optimizing Research Strategy for Next Loop ---")
    query = state["user_query"]
    notes = state.get("notes", [])
    compiled_notes = "\n\n".join(n for n in notes if n and n.strip()) if notes else "No notes gathered yet."
    structured_llm = planner_model.with_structured_output(ResearchPlan)
    messages = [
        SystemMessage(content="""You are a research gap analyst.
Review what has been gathered and identify what is MISSING by checking:
- Missing fundamentals (if basics like definition/types/causes are not yet covered, prioritise those)
- Missing perspectives (opposing views, expert disagreements)
- Missing data (statistics, named studies, primary sources)
- Missing depth (surface explanations need mechanisms/evidence)

Generate 2-4 NEW search queries that specifically fill these exact gaps.
Do NOT repeat or rephrase existing search queries — only generate genuinely new angles.
Do NOT anchor queries to a specific year unless the user asked for it."""),
        HumanMessage(content=f"ORIGINAL QUERY:\n{query}\n\nWHAT WE HAVE SO FAR:\n{compiled_notes}")
    ]
    try:
        plan_output = structured_llm.invoke(messages)
        new_plan = plan_output.steps
        print(f"New targeted search plan: {new_plan}")
    except Exception as e:
        print(f"Error optimizing research: {e}")
        new_plan = [query]
    return {"plan": new_plan}


def refine_query(state: ResearchState) -> ResearchState:
    """Sharpens gap-fill queries into precise, search-engine-optimised strings."""
    print("--- Refining Queries for Next Search Round ---")
    plan = state.get("plan", [])
    query = state.get("user_query", "")
    if not plan:
        return {}
    structured_llm = planner_model.with_structured_output(ResearchPlan)
    messages = [
        SystemMessage(content="""You are a search query optimizer.
Rewrite each query in the list to be more precise and search-engine-friendly.
Rules:
- Keep the core intent but use specific keywords, names, or technical terms
- Remove vague openers like "what is", "tell me about", "explain"
- Make each query distinct — no duplicates or near-duplicates allowed
- Keep each query under 10 words
Output only the refined list, nothing else."""),
        HumanMessage(content=f"ORIGINAL TOPIC: {query}\n\nQUERIES TO REFINE:\n" + "\n".join(f"- {q}" for q in plan))
    ]
    try:
        refined = structured_llm.invoke(messages)
        print(f"Refined queries: {refined.steps}")
        return {"plan": refined.steps}
    except Exception as e:
        print(f"[!] Query refinement failed: {e}. Keeping original plan.")
        return {}


def evaluation_router(state: ResearchState) -> str:
    """Routes to research_writer if approved, or optimize_research if more loops needed."""
    if state.get("is_complete", False):
        print("--- Route: Research Approved! ---")
        return "approved"
    if state["iteration"] < state["max_iteration"]:
        print("--- Route: Needs Improvement. Looping back... ---")
        return "needs_improvement"
    print("--- Route: Max iterations reached. Forcing Approval... ---")
    return "approved"


# Build and compile the graph
graph = StateGraph(ResearchState)

graph.add_node("user_query",       user_query_node)
graph.add_node("web_search",       web_search)
graph.add_node("extract_info",     extract_info)
graph.add_node("summarize_notes",  summarize_notes)
graph.add_node("evaluate",         evaluate)
graph.add_node("optimize_research", optimize_research)
graph.add_node("refine_query",     refine_query)
graph.add_node("research_writer",  research_writer)
graph.add_node("save_notes",       save_notes)

graph.add_edge(START,             "user_query")
graph.add_edge("user_query",      "web_search")
graph.add_edge("web_search",      "extract_info")
graph.add_edge("extract_info",    "summarize_notes")
graph.add_edge("summarize_notes", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    evaluation_router,
    {
        "approved":         "research_writer",
        "needs_improvement": "optimize_research",
    }
)

graph.add_edge("optimize_research", "refine_query")
graph.add_edge("refine_query",      "web_search")
graph.add_edge("research_writer",   "save_notes")
graph.add_edge("save_notes",        END)

app = graph.compile()


if __name__ == "__main__":
    print("==================================================")
    print("      🚀 AI RESEARCH ASSISTANT ACTIVATED 🚀       ")
    print("==================================================")

    user_query = input("\nEnter your research question, prompt, or topic:\n> ").strip()
    if not user_query:
        print("\n[!] No input provided. Defaulting to 'What is LangGraph?'")
        user_query = "What is LangGraph?"

    initial_state = {
        "user_query": user_query,
        "target_language": "English",
        "writing_style": "Book-Style Detailed",
        "target_length": "Medium-depth (3-4 paragraphs)",
        "iteration": 0,
        "max_iteration": 3,
        "web_results": "",
        "web_links": [],
        "final_answer": "",
        "plan": [],
        "notes": [],
        "is_complete": False,
    }

    print(f"\n[+] Starting research on: '{user_query}'...")
    print("Running research graph (this may take a few moments)...\n")

    result = app.invoke(initial_state)
    final_report = result["final_answer"]

    print("\n==================================================")
    print("              📝 FINAL RESEARCH REPORT             ")
    print("==================================================")
    print(final_report)
    print("==================================================")

    print("\nHow would you like to save this report?")
    print("  [1] Markdown (.md)  - Best for Notion, Obsidian, VS Code")
    print("  [2] Plain Text (.txt) - Opens on any device")
    print("  [3] PDF (.pdf)       - Best for sharing and printing")
    print("  [4] Skip            - Don't save")
    fmt_choice = input("\nEnter your choice (1/2/3/4) [default: 1]: ").strip()

    format_map = {"1": "md", "2": "txt", "3": "pdf", "4": None, "": "md"}
    output_format = format_map.get(fmt_choice, "md")

    if output_format is None:
        print("\n[~] Skipped saving. Goodbye!")
    else:
        query_slug = re.sub(r'[^\w\s-]', '', user_query)[:40].strip().replace(" ", "_")
        filename = f"research_report_{query_slug}"

        if output_format == "txt":
            filepath = f"{filename}.txt"
            plain = re.sub(r'#{1,6}\s?', '', final_report)
            plain = re.sub(r'\*\*(.*?)\*\*', r'\1', plain)
            plain = re.sub(r'\*(.*?)\*', r'\1', plain)
            with open(filepath, "w", encoding="utf-8-sig") as f:
                f.write(plain)
            print(f"[✓] Report saved as Plain Text: {filepath}")

        elif output_format == "pdf":
            filepath = f"{filename}.pdf"
            if not PDF_AVAILABLE:
                print("[!] fpdf2 not installed. Run: pip install fpdf2")
                print("[~] Falling back to Markdown.")
                filepath = f"{filename}.md"
                with open(filepath, "w", encoding="utf-8-sig") as f:
                    f.write(final_report)
                print(f"[✓] Report saved as Markdown: {filepath}")
            else:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()

                def render_pdf_line(pdf, line):
                    line = line.rstrip()
                    if not line:
                        pdf.ln(3)
                        return
                    if line.startswith("# "):
                        pdf.set_font("Helvetica", "B", 16)
                        pdf.multi_cell(0, 10, line[2:])
                        pdf.ln(2)
                    elif line.startswith("## "):
                        pdf.set_font("Helvetica", "B", 14)
                        pdf.multi_cell(0, 9, line[3:])
                        pdf.ln(2)
                    elif line.startswith("### "):
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.multi_cell(0, 8, line[4:])
                        pdf.ln(1)
                    elif line.startswith("---"):
                        pdf.set_draw_color(180, 180, 180)
                        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
                        pdf.ln(4)
                    elif line.startswith("- ") or line.startswith("* "):
                        pdf.set_font("Helvetica", "", 11)
                        pdf.set_x(15)
                        pdf.multi_cell(0, 7, f"• {line[2:]}")
                    else:
                        text = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                        text = re.sub(r'\*(.*?)\*', r'\1', text)
                        text = re.sub(r'`(.*?)`', r'\1', text)
                        pdf.set_font("Helvetica", "", 11)
                        pdf.multi_cell(0, 7, text)

                for line in final_report.split("\n"):
                    try:
                        render_pdf_line(pdf, line)
                    except Exception:
                        pass
                pdf.output(filepath)
                print(f"[✓] Report saved as PDF: {filepath}")

        else:
            filepath = f"{filename}.md"
            with open(filepath, "w", encoding="utf-8-sig") as f:
                f.write(final_report)
            print(f"[✓] Report saved as Markdown: {filepath}")

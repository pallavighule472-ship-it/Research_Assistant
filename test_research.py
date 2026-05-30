"""
test_research.py — Research Assistant test suite

Run:
    pytest test_research.py -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Must be set before any LangChain import (no network call at init, but key is validated)
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing-only")
os.environ.pop("TAVILY_API_KEY", None)   # ensure Tavily is disabled in all tests


# ─── Shared fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rb():
    """Import Research_Backend once per test session (module-level init runs once)."""
    import Research_Backend
    return Research_Backend


# ─── 1. SKIP_DOMAINS ──────────────────────────────────────────────────────────

class TestSkipDomains:
    def test_social_and_video_domains_present(self, rb):
        for domain in ("youtube.com", "reddit.com", "facebook.com", "twitter.com", "x.com", "instagram.com"):
            assert domain in rb.SKIP_DOMAINS, f"{domain} missing from SKIP_DOMAINS"

    def test_skip_domain_urls_are_filtered(self, rb):
        urls = [
            "https://youtube.com/watch",
            "https://reddit.com/r/test",
            "https://schema.org/Thing",
        ]
        result = [u for u in urls if not any(d in u for d in rb.SKIP_DOMAINS)]
        assert result == []

    def test_quality_domain_passes_filter(self, rb):
        for url in ("https://arxiv.org/abs/2301.00001", "https://nature.com/articles/test"):
            assert not any(d in url for d in rb.SKIP_DOMAINS)


# ─── 2. crawl_webpage ─────────────────────────────────────────────────────────

class TestCrawlWebpage:
    def test_rejects_non_http(self, rb):
        assert rb.crawl_webpage("ftp://example.com") == ""

    def test_rejects_binary_extensions(self, rb):
        for ext in (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".mp4"):
            assert rb.crawl_webpage(f"https://example.com/file{ext}") == ""

    def test_rejects_skip_domain_urls(self, rb):
        assert rb.crawl_webpage("https://www.youtube.com/watch?v=test") == ""
        assert rb.crawl_webpage("https://reddit.com/r/python") == ""

    def test_returns_text_on_200(self, rb):
        body = "Meaningful content about LangGraph here. " * 10
        mock_resp = MagicMock(status_code=200, text=f"<html><body><p>{body}</p></body></html>")
        with patch("Research_Backend.requests.get", return_value=mock_resp):
            result = rb.crawl_webpage("https://example.com/article")
        assert len(result) > 0

    def test_returns_empty_on_non_200(self, rb):
        mock_resp = MagicMock(status_code=404)
        with patch("Research_Backend.requests.get", return_value=mock_resp):
            assert rb.crawl_webpage("https://example.com/missing") == ""

    def test_returns_empty_on_network_error(self, rb):
        with patch("Research_Backend.requests.get", side_effect=Exception("connection refused")):
            assert rb.crawl_webpage("https://example.com/broken") == ""


# ─── 3. evaluation_router ─────────────────────────────────────────────────────

class TestEvaluationRouter:
    def test_approved_when_complete(self, rb):
        assert rb.evaluation_router({"is_complete": True, "iteration": 1, "max_iteration": 3}) == "approved"

    def test_needs_improvement_when_loops_remain(self, rb):
        assert rb.evaluation_router({"is_complete": False, "iteration": 1, "max_iteration": 3}) == "needs_improvement"

    def test_forces_approved_at_max_iteration(self, rb):
        assert rb.evaluation_router({"is_complete": False, "iteration": 3, "max_iteration": 3}) == "approved"

    def test_forces_approved_with_depth_one(self, rb):
        assert rb.evaluation_router({"is_complete": False, "iteration": 1, "max_iteration": 1}) == "approved"


# ─── 4. user_query_node ───────────────────────────────────────────────────────

class TestUserQueryNode:
    _BASE = {
        "user_query": "What is LangGraph?",
        "target_language": "English",
        "writing_style": "Book-Style Detailed",
        "target_length": "Medium-depth (3-4 paragraphs)",
        "plan": [], "notes": [], "iteration": 0, "max_iteration": 3,
        "web_results": "", "web_links": [], "final_answer": "", "is_complete": False,
    }

    def test_returns_llm_plan(self, rb):
        from Research_Backend import ResearchPlan
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ResearchPlan(
            steps=["Overview of LangGraph", "Key components", "Use cases"]
        )
        with patch("Research_Backend.planner_model", mock_llm):
            result = rb.user_query_node(self._BASE)
        assert result["plan"] == ["Overview of LangGraph", "Key components", "Use cases"]
        assert result["iteration"] == 0
        assert result["notes"] == []

    def test_falls_back_to_default_plan_on_llm_error(self, rb):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")
        with patch("Research_Backend.planner_model", mock_llm):
            result = rb.user_query_node(self._BASE)
        assert len(result["plan"]) >= 3

    def test_degenerate_plan_replaced_with_default(self, rb):
        """A plan whose only item mirrors the query exactly is rejected and replaced."""
        from Research_Backend import ResearchPlan
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ResearchPlan(
            steps=["What is LangGraph?"]
        )
        with patch("Research_Backend.planner_model", mock_llm):
            result = rb.user_query_node(self._BASE)
        assert len(result["plan"]) > 1


# ─── 5. evaluate node ─────────────────────────────────────────────────────────

class TestEvaluateNode:
    def _state(self, iteration=0, max_iteration=3):
        return {
            "user_query": "What is LangGraph?",
            "notes": ["Research notes about LangGraph state graphs and nodes."],
            "iteration": iteration, "max_iteration": max_iteration,
            "target_language": "English", "writing_style": "Book-Style Detailed",
            "target_length": "Medium-depth (3-4 paragraphs)",
            "plan": [], "web_results": "", "web_links": [], "final_answer": "", "is_complete": False,
        }

    def test_marks_complete_when_llm_says_so(self, rb):
        from Research_Backend import ResearchEvaluation
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ResearchEvaluation(
            is_complete=True, reasoning="All criteria met."
        )
        with patch("Research_Backend.evaluate_model", mock_llm):
            result = rb.evaluate(self._state())
        assert result["is_complete"] is True
        assert result["iteration"] == 1

    def test_increments_iteration_on_incomplete(self, rb):
        from Research_Backend import ResearchEvaluation
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ResearchEvaluation(
            is_complete=False, reasoning="Needs more depth."
        )
        with patch("Research_Backend.evaluate_model", mock_llm):
            result = rb.evaluate(self._state(iteration=1))
        assert result["iteration"] == 2

    def test_iteration_capped_at_max(self, rb):
        from Research_Backend import ResearchEvaluation
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ResearchEvaluation(
            is_complete=False, reasoning="Still incomplete."
        )
        with patch("Research_Backend.evaluate_model", mock_llm):
            result = rb.evaluate(self._state(iteration=3, max_iteration=3))
        assert result["iteration"] == 3  # must not exceed max_iteration


# ─── 6. Full graph integration test ───────────────────────────────────────────

class TestGraphIntegration:
    @staticmethod
    def _msg(content: str):
        m = MagicMock()
        m.content = content
        return m

    def test_full_graph_cycle_produces_report(self, rb):
        """
        Runs the compiled LangGraph app end-to-end with all external calls mocked.
        Validates that a non-empty final_answer is produced and is_complete is True.
        """
        from Research_Backend import ResearchPlan, ResearchEvaluation

        mock_planner = MagicMock()
        mock_planner.with_structured_output.return_value.invoke.return_value = ResearchPlan(
            steps=["LangGraph overview", "LangGraph components"]
        )

        mock_evaluator = MagicMock()
        mock_evaluator.with_structured_output.return_value.invoke.return_value = ResearchEvaluation(
            is_complete=True, reasoning="Sufficient for a 1-iteration run."
        )
        mock_evaluator.invoke.return_value = self._msg("Compressed: LangGraph orchestrates LLM agents via directed graphs.")

        mock_extractor = MagicMock()
        mock_extractor.invoke.return_value = self._msg(
            "- LangGraph uses StateGraph for orchestration.\n- Nodes are Python functions."
        )

        # Mock DDGS context manager: DDGS() -> cm -> cm.__enter__() -> ddgs_obj
        mock_ddgs_obj = MagicMock()
        mock_ddgs_obj.text.return_value = [{"body": "LangGraph is a stateful LLM orchestration framework."}]
        mock_ddgs_cm = MagicMock()
        mock_ddgs_cm.__enter__ = MagicMock(return_value=mock_ddgs_obj)
        mock_ddgs_cm.__exit__ = MagicMock(return_value=False)
        mock_DDGS = MagicMock(return_value=mock_ddgs_cm)

        # Mock wikipedia package
        mock_wiki_pkg = MagicMock()
        mock_wiki_pkg.summary.return_value = "LangGraph extends LangChain with graph control flow."
        mock_wiki_pkg.exceptions.DisambiguationError = Exception

        mock_http = MagicMock(
            status_code=200,
            text="<html><body><p>" + "LangGraph content. " * 60 + "</p></body></html>",
        )

        mock_writer = MagicMock()
        mock_writer.invoke.return_value = self._msg(
            "## LangGraph overview\n\nLangGraph is a graph-based agent orchestration framework."
        )

        initial_state = {
            "user_query": "What is LangGraph?",
            "target_language": "English",
            "writing_style": "Book-Style Detailed",
            "target_length": "Medium-depth (3-4 paragraphs)",
            "iteration": 0,
            "max_iteration": 1,
            "web_results": "",
            "web_links": [],
            "final_answer": "",
            "plan": [],
            "notes": [],
            "is_complete": False,
        }

        with patch.multiple(
            "Research_Backend",
            planner_model=mock_planner,
            evaluate_model=mock_evaluator,
            extract_model=mock_extractor,
            DDGS=mock_DDGS,
            _wiki_pkg=mock_wiki_pkg,
            tavily_search=None,
            ChatOpenAI=MagicMock(return_value=mock_writer),
        ), patch("Research_Backend.requests.get", return_value=mock_http), \
           patch("Research_Backend.time.sleep"):
            result = rb.app.invoke(initial_state)

        assert "final_answer" in result
        assert len(result["final_answer"]) > 100
        assert result["is_complete"] is True


# ─── 7. _append_references ───────────────────────────────────────────────────

class TestAppendReferences:
    def test_appends_references_when_absent(self, rb):
        draft = "## Conclusion\n\nDone."
        links = ["https://arxiv.org/abs/1234", "https://nature.com/article"]
        result = rb._append_references(draft, links)
        assert "## References" in result
        assert "arxiv.org" in result
        assert "nature.com" in result

    def test_replaces_existing_references_with_numbered_format(self, rb):
        draft = "## References\n\n[1] https://arxiv.org\n\n"
        links = ["https://arxiv.org/abs/1234"]
        result = rb._append_references(draft, links)
        assert result.count("## References") == 1
        assert "Available at: https://arxiv.org/abs/1234" in result

    def test_filters_skip_domain_links(self, rb):
        draft = "## Conclusion\n\nDone."
        links = ["https://youtube.com/watch?v=abc", "https://reddit.com/r/ml"]
        result = rb._append_references(draft, links)
        assert "## References" not in result

    def test_mixed_links_only_quality_kept(self, rb):
        draft = "## Conclusion\n\nDone."
        links = ["https://youtube.com/watch", "https://arxiv.org/abs/9999"]
        result = rb._append_references(draft, links)
        assert "arxiv.org" in result
        assert "youtube.com" not in result

    def test_no_references_on_empty_links(self, rb):
        draft = "## Conclusion\n\nDone."
        result = rb._append_references(draft, [])
        assert "## References" not in result


# ─── 9. FastAPI endpoint tests ────────────────────────────────────────────────

class TestFastAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx not installed — run: pip install httpx")
        from Research_Backend_Server import server
        return TestClient(server)

    def test_health_returns_200_and_healthy_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_redirects_to_docs(self, client):
        from fastapi.testclient import TestClient
        from Research_Backend_Server import server
        no_follow = TestClient(server, follow_redirects=False)
        assert no_follow.get("/").status_code in (301, 302, 307, 308)


# ─── 10. Request payload validation ───────────────────────────────────────────

class TestRequestValidation:
    def test_valid_max_iteration_accepted(self):
        from Research_Backend_Server import ResearchRequest
        assert ResearchRequest(query="test", max_iteration=3).max_iteration == 3

    def test_default_values(self):
        from Research_Backend_Server import ResearchRequest
        req = ResearchRequest(query="test")
        assert req.target_language == "English"
        assert req.writing_style == "Book-Style Detailed"
        assert req.max_iteration == 3

    def test_max_iteration_zero_rejected(self):
        from pydantic import ValidationError
        from Research_Backend_Server import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(query="test", max_iteration=0)

    def test_max_iteration_six_rejected(self):
        from pydantic import ValidationError
        from Research_Backend_Server import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(query="test", max_iteration=6)

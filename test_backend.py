import unittest

from pollutant_catalog import infer_industry_hint, parse_measurements
from standards import evaluate_records
from law_query_categories import build_category_query, normalize_category
from backend_service import (
    build_structured_judgment,
    canonicalize_citation,
    dedupe_citations,
    memory_store,
    process_query,
)


class BackendParsingTests(unittest.TestCase):
    def test_parse_pollutant_value_unit_after_alias(self):
        self.assertEqual(
            parse_measurements("我們工廠廢水 COD 測到 120 mg/L，這樣合規嗎？"),
            {"pollutant": "COD", "value": 120.0, "unit": "mg/L"},
        )

    def test_short_english_alias_does_not_match_inside_longer_token(self):
        self.assertEqual(
            parse_measurements("半導體 COD 120 mg/L"),
            {"pollutant": "COD", "value": 120.0, "unit": "mg/L"},
        )

    def test_parse_ph_without_unit_does_not_capture_question_text(self):
        self.assertEqual(
            parse_measurements("pH 5.8 合規嗎？"),
            {"pollutant": "pH", "value": 5.8, "unit": ""},
        )

    def test_parse_chinese_alias_and_unit(self):
        self.assertEqual(
            parse_measurements("懸浮固體 20 毫克/公升"),
            {"pollutant": "SS", "value": 20.0, "unit": "mg/L"},
        )

    def test_parse_value_before_alias(self):
        self.assertEqual(
            parse_measurements("120 mg/L COD"),
            {"pollutant": "COD", "value": 120.0, "unit": "mg/L"},
        )


class StructuredStandardTests(unittest.TestCase):
    def test_industry_hint_required_for_semiconductor_standard(self):
        result = evaluate_records({"pollutant": "COD", "value": 120.0, "unit": "mg/L"})
        self.assertEqual(result[0]["status"], "no_standard")

    def test_semiconductor_cod_limit_fails_when_over_limit(self):
        result = evaluate_records(
            {"pollutant": "COD", "value": 120.0, "unit": "mg/L"},
            industry_hint="晶圓製造及半導體製造業",
        )
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[0]["standard"]["limit_value"], 100)

    def test_semiconductor_ph_range_fails_when_out_of_range(self):
        result = evaluate_records(
            {"pollutant": "pH", "value": 5.8, "unit": ""},
            industry_hint="晶圓製造及半導體製造業",
        )
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[0]["standard"]["limit_type"], "range")

    def test_industry_hint_from_alias(self):
        self.assertEqual(
            infer_industry_hint("半導體 COD 120 mg/L 合規嗎？"),
            "晶圓製造及半導體製造業",
        )

    def test_required_condition_blocks_direct_comparison(self):
        standards = [{
            "id": "conditional_nh3n",
            "industry": "測試產業",
            "pollutant": "NH3-N",
            "aliases": ["氨氮"],
            "limit_type": "max",
            "limit_value": 10,
            "unit": "mg/L",
            "requires_industry": True,
            "conditions": [{
                "field": "protection_area",
                "label": "是否位於自來水水質水量保護區",
                "operator": "equals",
                "value": True,
                "required": True,
            }],
        }]

        result = evaluate_records(
            {"pollutant": "NH3-N", "value": 12.0, "unit": "mg/L"},
            industry_hint="測試產業",
            standards=standards,
        )
        self.assertEqual(result[0]["status"], "missing_conditions")
        self.assertEqual(result[0]["missing_conditions"][0]["field"], "protection_area")


class BackendApiCategoryTests(unittest.TestCase):
    def test_unknown_category_falls_back_to_auto(self):
        self.assertEqual(normalize_category("unknown"), "auto")

    def test_auto_category_keeps_query_unchanged(self):
        self.assertEqual(build_category_query("COD 120 mg/L", "auto"), "COD 120 mg/L")

    def test_specific_category_adds_context(self):
        result = build_category_query("COD 120 mg/L", "effluent_standard")
        self.assertIn("放流水標準", result)
        self.assertIn("COD 120 mg/L", result)


class BackendServiceTests(unittest.TestCase):
    def setUp(self):
        memory_store.clear()

    def test_missing_industry_triggers_data_request_for_compliance_query(self):
        payload = process_query(
            query="COD 120 mg/L 是否符合放流水標準",
            conversation_id="c1",
            category="effluent_standard",
            run_graph=lambda query: {"final_answer": "should not run"},
        )
        self.assertTrue(payload["waiting_for_data_input"])
        self.assertIn("適用對象", payload["data_request_hint"])

    def test_structured_judgment_returns_fixed_schema(self):
        structured = build_structured_judgment(
            measurements={"pollutant": "COD", "value": 120.0, "unit": "mg/L"},
            industry_hint="化工業",
            category="effluent_standard",
        )
        self.assertIsNotNone(structured)
        self.assertIn(structured["overall_status"], {"passed", "failed", "no_standard", "missing_data"})
        self.assertIn("items", structured)
        self.assertIn("pollutant", structured["items"][0])
        self.assertIn("source_article", structured["items"][0])

    def test_conversation_memory_enriches_followup_query(self):
        captured = []

        def fake_graph(query):
            captured.append(query)
            return {
                "final_answer": "ok",
                "law_search_results": [],
                "cited_articles": [],
            }

        first = process_query(
            query="化工業 COD 120 mg/L 是否符合放流水標準",
            conversation_id="c2",
            category="effluent_standard",
            run_graph=fake_graph,
        )
        self.assertFalse(first["waiting_for_data_input"])

        second = process_query(
            query="那如果是 pH 5 呢",
            conversation_id="c2",
            category="effluent_standard",
            run_graph=fake_graph,
        )
        self.assertFalse(second["waiting_for_data_input"])
        self.assertIn("上一題", captured[-1])
        self.assertIn("化工業", captured[-1])

    def test_structured_result_falls_back_when_graph_fails(self):
        payload = process_query(
            query="化工業 COD 120 mg/L 是否符合放流水標準",
            conversation_id="c3",
            category="effluent_standard",
            run_graph=lambda query: (_ for _ in ()).throw(RuntimeError("graph unavailable")),
        )
        self.assertIn("structured_judgment", payload)
        self.assertIsNotNone(payload["structured_judgment"])
        self.assertIn("Graph fallback used", payload["error"])

    def test_industry_scope_summary_does_not_call_graph(self):
        called = []
        payload = process_query(
            query="\u4e3b\u8981\u6cd5\u898f\u6709\u54ea\u4e9b",
            conversation_id="c4",
            category="industry_scope",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("\u9644\u8868\u4e00", payload["final_answer"])
        self.assertIn("\u9644\u8868\u516b", payload["final_answer"])
        self.assertIn("\u5316\u5de5\u696d", payload["final_answer"])
        self.assertEqual(payload["citations"][0]["type"], "category_summary")

    def test_facility_special_summary_does_not_call_graph(self):
        called = []
        payload = process_query(
            query="\u4e3b\u8981\u6cd5\u898f\u6709\u54ea\u4e9b",
            conversation_id="c5",
            category="facility_special",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("\u9644\u8868\u5341\u4e94", payload["final_answer"])
        self.assertIn("\u9644\u8868\u5341\u516d", payload["final_answer"])
        self.assertIn("\u5efa\u7bc9\u7269\u6c61\u6c34\u8655\u7406\u8a2d\u65bd", payload["final_answer"])

    def test_sewer_system_summary_does_not_call_graph(self):
        called = []
        payload = process_query(
            query="\u4e3b\u8981\u6cd5\u898f\u6709\u54ea\u4e9b",
            conversation_id="c6",
            category="sewer_system",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("\u9644\u8868\u4e5d", payload["final_answer"])
        self.assertIn("\u9644\u8868\u5341\u56db", payload["final_answer"])

    def test_effluent_standard_numeric_query_still_uses_structured_flow(self):
        called = []
        payload = process_query(
            query="\u5316\u5de5\u696d COD 120 mg/L \u662f\u5426\u7b26\u5408\u653e\u6d41\u6c34\u6a19\u6e96",
            conversation_id="c7",
            category="effluent_standard",
            run_graph=lambda query: called.append(query) or {
                "final_answer": "graph answer",
                "law_search_results": [],
                "cited_articles": [],
            },
        )
        self.assertNotEqual(called, [])
        self.assertIsNotNone(payload["structured_judgment"])

    def test_dedupe_citations_drops_bare_article_when_full_title_exists(self):
        citations = dedupe_citations([
            {"title": "\u7b2c 35 \u689d", "text": ""},
            {
                "title": "\u6c34\u6c61\u67d3\u9632\u6cbb\u63aa\u65bd\u8a08\u756b\u53ca\u8a31\u53ef\u7533\u8acb\u5be9\u67e5\u7ba1\u7406\u8fa6\u6cd5 \u7b2c 35 \u689d",
                "text": "",
            },
        ])
        self.assertEqual(len(citations), 1)
        self.assertIn("\u6c34\u6c61\u67d3\u9632\u6cbb\u63aa\u65bd", citations[0]["title"])

    def test_dedupe_citations_keeps_same_article_from_different_laws(self):
        citations = dedupe_citations([
            {"title": "\u6c34\u6c61\u67d3\u9632\u6cbb\u6cd5 \u7b2c 35 \u689d", "text": ""},
            {
                "title": "\u6c34\u6c61\u67d3\u9632\u6cbb\u63aa\u65bd\u8a08\u756b\u53ca\u8a31\u53ef\u7533\u8acb\u5be9\u67e5\u7ba1\u7406\u8fa6\u6cd5 \u7b2c 35 \u689d",
                "text": "",
            },
        ])
        self.assertEqual(len(citations), 2)

    def test_dedupe_citations_drops_ambiguous_bare_article(self):
        citations = dedupe_citations([
            {"title": "\u7b2c 35 \u689d", "text": "\u88f8\u689d\u6587"},
            {"title": "\u6c34\u6c61\u67d3\u9632\u6cbb\u6cd5 \u7b2c 35 \u689d", "text": ""},
            {
                "title": "\u6c34\u6c61\u67d3\u9632\u6cbb\u63aa\u65bd\u8a08\u756b\u53ca\u8a31\u53ef\u7533\u8acb\u5be9\u67e5\u7ba1\u7406\u8fa6\u6cd5 \u7b2c 35 \u689d",
                "text": "",
            },
        ])
        self.assertEqual(len(citations), 2)
        self.assertNotIn("\u7b2c 35 \u689d", [item["title"] for item in citations])

    def test_canonicalize_citation_parses_article_and_table(self):
        article = canonicalize_citation({"title": "\u7b2c 35 \u689d"})
        table = canonicalize_citation({"title": "\u9644\u8868\u516d"})
        full_table = canonicalize_citation({
            "title": "\u9644\u8868\u516d\u767c\u96fb\u5ee0\u653e\u6d41\u6c34\u6c34\u8cea\u9805\u76ee\u53ca\u9650\u503c"
        })
        self.assertEqual(article["kind"], "article")
        self.assertEqual(article["article_no"], "35")
        self.assertEqual(table["kind"], "table")
        self.assertEqual(table["table_no"], "\u516d")
        self.assertGreater(full_table["specificity_score"], table["specificity_score"])

    def test_dedupe_citations_drops_bare_table_when_full_title_exists(self):
        citations = dedupe_citations([
            {"title": "\u9644\u8868\u516d", "text": "\u88dc\u5145\u6587\u5b57"},
            {
                "title": "\u9644\u8868\u516d\u767c\u96fb\u5ee0\u653e\u6d41\u6c34\u6c34\u8cea\u9805\u76ee\u53ca\u9650\u503c",
                "text": "",
            },
        ])
        self.assertEqual(len(citations), 1)
        self.assertIn("\u767c\u96fb\u5ee0", citations[0]["title"])
        self.assertEqual(citations[0]["text"], "\u88dc\u5145\u6587\u5b57")

    def test_dedupe_citations_keeps_different_tables(self):
        citations = dedupe_citations([
            {"title": "\u9644\u8868\u516d", "text": ""},
            {"title": "\u9644\u8868\u5341\u516d", "text": ""},
        ])
        self.assertEqual(len(citations), 2)


if __name__ == "__main__":
    unittest.main()

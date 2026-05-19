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

    def test_pollutant_formula_number_is_not_measurement(self):
        self.assertIsNone(
            parse_measurements("光電材料及元件製造業排放廢水時，氨氮 NH3-N 的標準是多少？")
        )
        self.assertIsNone(parse_measurements("BOD5 標準是多少？"))

    def test_parse_real_measurement_after_formula_alias(self):
        self.assertEqual(
            parse_measurements("光電材料及元件製造業 NH3-N 12 mg/L 是否合規？"),
            {"pollutant": "NH3-N", "value": 12.0, "unit": "mg/L"},
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

    def test_semiconductor_ph_limit_query_uses_structured_standard(self):
        called = []
        payload = process_query(
            query="\u6676\u5713\u88fd\u9020\u53ca\u534a\u5c0e\u9ad4\u88fd\u9020\u696d\u7684\u653e\u6d41\u6c34 pH \u9650\u503c\u662f\u591a\u5c11\uff1f",
            conversation_id="limit1",
            category="effluent_standard",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("6", payload["final_answer"])
        self.assertIn("9", payload["final_answer"])
        self.assertIn("\u7121\u55ae\u4f4d", payload["final_answer"])
        self.assertIn("\u9644\u8868\u4e00", payload["final_answer"])
        self.assertNotIn("\u8cc7\u6599\u4e0d\u8db3", payload["final_answer"])
        self.assertEqual(payload["citations"][0]["type"], "structured_limit_query")
        self.assertNotIn("\u7b2c 1 \u689d", [item["title"] for item in payload["citations"]])
        self.assertNotIn("\u7b2c 44 \u689d", [item["title"] for item in payload["citations"]])

    def test_chemical_cod_limit_query_uses_structured_standard(self):
        called = []
        payload = process_query(
            query="\u5316\u5de5\u696d COD \u9650\u503c\u662f\u591a\u5c11\uff1f",
            conversation_id="limit2",
            category="effluent_standard",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("COD", payload["final_answer"])
        self.assertIn("mg/L", payload["final_answer"])
        self.assertNotIn("\u8cc7\u6599\u4e0d\u8db3", payload["final_answer"])

    def test_electroplating_ss_limit_query_uses_alias(self):
        called = []
        payload = process_query(
            query="\u96fb\u934d\u696d\u61f8\u6d6e\u56fa\u9ad4\u6a19\u6e96\u662f\u591a\u5c11\uff1f",
            conversation_id="limit3",
            category="effluent_standard",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("SS", payload["final_answer"])
        self.assertIn("mg/L", payload["final_answer"])
        self.assertNotIn("\u8cc7\u6599\u4e0d\u8db3", payload["final_answer"])

    def test_science_park_sewer_ph_limit_query_uses_sewer_scope(self):
        called = []
        payload = process_query(
            query="\u79d1\u5b78\u5de5\u696d\u5712\u5340\u5c08\u7528\u6c61\u6c34\u4e0b\u6c34\u9053\u7cfb\u7d71 pH \u6a19\u6e96\uff1f",
            conversation_id="limit4",
            category="sewer_system",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("6", payload["final_answer"])
        self.assertIn("9", payload["final_answer"])
        self.assertIn("\u9644\u8868\u4e5d", payload["final_answer"])

    def test_limit_query_does_not_intercept_numeric_compliance_query(self):
        called = []
        payload = process_query(
            query="\u6676\u5713\u88fd\u9020\u53ca\u534a\u5c0e\u9ad4\u88fd\u9020\u696d pH 5.8 \u662f\u5426\u5408\u898f\uff1f",
            conversation_id="limit5",
            category="effluent_standard",
            run_graph=lambda query: called.append(query) or {
                "final_answer": "graph answer",
                "law_search_results": [],
                "cited_articles": [],
            },
        )
        self.assertNotEqual(called, [])
        self.assertIsNotNone(payload["structured_judgment"])
        self.assertEqual(payload["structured_judgment"]["overall_status"], "failed")

    def test_conditional_limit_query_requests_missing_conditions(self):
        called = []
        payload = process_query(
            query="\u79d1\u5b78\u5de5\u696d\u5712\u5340\u5c08\u7528\u6c61\u6c34\u4e0b\u6c34\u9053\u7cfb\u7d71 \u6c28\u6c2e \u9650\u503c\u662f\u591a\u5c11\uff1f",
            conversation_id="limit6",
            category="sewer_system",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("\u4f9d\u9069\u7528\u60c5\u5883\u4e0d\u540c", payload["final_answer"])
        self.assertIn("10 mg/L", payload["final_answer"])
        self.assertIn("20 mg/L", payload["final_answer"])
        self.assertIn("30 mg/L", payload["final_answer"])
        self.assertNotIn("\u9700\u88dc\u5145", payload["final_answer"])

    def test_optoelectronics_nh3n_limit_query_lists_conditional_limits(self):
        called = []
        payload = process_query(
            query="\u5149\u96fb\u6750\u6599\u53ca\u5143\u4ef6\u88fd\u9020\u696d\u6392\u653e\u5ee2\u6c34\u6642\uff0c\u6c28\u6c2e NH3-N \u7684\u6a19\u6e96\u662f\u591a\u5c11\uff1f",
            conversation_id="limit7",
            category="auto",
            run_graph=lambda query: called.append(query) or {
                "final_answer": "wrong",
                "law_search_results": [{"law_name": "\u6c34\u6c61\u67d3\u9632\u6cbb\u6cd5", "article": "\u7b2c 19 \u689d", "text": "wrong"}],
                "cited_articles": ["\u6c34\u6c61\u67d3\u9632\u6cbb\u6cd5 \u7b2c 19 \u689d"],
            },
        )
        self.assertEqual(called, [])
        self.assertIn("10 mg/L", payload["final_answer"])
        self.assertIn("20 mg/L", payload["final_answer"])
        self.assertIn("30 mg/L", payload["final_answer"])
        self.assertIn("\u9644\u8868\u4e8c", payload["final_answer"])
        self.assertIn("\u4f9d\u9069\u7528\u60c5\u5883\u4e0d\u540c", payload["final_answer"])
        self.assertNotIn("\u9700\u8981\u88dc\u5145\u9069\u7528\u689d\u4ef6\u5f8c\u624d\u80fd\u5b8c\u6210\u5224\u65b7", payload["final_answer"])
        titles = [item["title"] for item in payload["citations"]]
        joined_titles = "\n".join(titles)
        self.assertIn("\u9644\u8868\u4e8c", joined_titles)
        self.assertNotIn("\u6c34\u6c61\u67d3\u9632\u6cbb\u6cd5", joined_titles)
        self.assertNotIn("\u6c34\u6c61\u67d3\u9632\u6cbb\u63aa\u65bd", joined_titles)
        self.assertNotIn("\u9644\u8868\u516b", joined_titles)

    def test_metal_surface_heavy_metal_limit_query_uses_table_five(self):
        called = []
        payload = process_query(
            query="金屬表面處理業的銅、鎳、鋅、總鉻限值是多少？",
            conversation_id="limit8",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("3 mg/L", payload["final_answer"])
        self.assertIn("1 mg/L", payload["final_answer"])
        self.assertIn("5 mg/L", payload["final_answer"])
        self.assertIn("2 mg/L", payload["final_answer"])
        self.assertIn("\u9644\u8868\u4e94", payload["final_answer"])
        self.assertIn("六價鉻", payload["final_answer"])
        titles = "\n".join(item["title"] for item in payload["citations"])
        self.assertIn("\u9644\u8868\u4e94", titles)
        self.assertNotIn("\u9644\u8868\u516b", titles)

    def test_electroplating_hexavalent_chromium_uses_table_five(self):
        called = []
        payload = process_query(
            query="電鍍業六價鉻限值是多少？",
            conversation_id="limit9",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("0.5 mg/L", payload["final_answer"])
        titles = "\n".join(item["title"] for item in payload["citations"])
        self.assertIn("\u9644\u8868\u4e94", titles)
        self.assertNotIn("\u9644\u8868\u516b", titles)

    def test_other_industry_copper_still_uses_table_eight(self):
        called = []
        payload = process_query(
            query="\u5176\u4ed6\u4e8b\u696d\u9285\u9650\u503c\u662f\u591a\u5c11\uff1f",
            conversation_id="limit10",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("3 mg/L", payload["final_answer"])
        titles = "\n".join(item["title"] for item in payload["citations"])
        self.assertIn("\u9644\u8868\u516b", titles)

    def test_water_regression_dataset_routes_and_citations(self):
        cases = [
            {
                "question": "印刷電路板製造業排放含重金屬廢水，要看哪些水質項目？",
                "route": "item_list_query",
                "expected": ["重金屬", "銅", "鎳", "鋅", "總鉻", "六價鉻", "附表五"],
                "forbidden": ["資料不足"],
                "expected_citations": ["附表五"],
                "forbidden_citations": ["附表八"],
            },
            {
                "question": "我是半導體封裝測試廠，應該適用晶圓製造及半導體製造業的放流水標準嗎？",
                "route": "scope_query",
                "expected": ["不能只因", "半導體", "確認實際製程", "排放去向"],
                "forbidden": ["資料不足"],
                "expected_citations": ["附表一"],
            },
            {
                "question": "我們是位於科學園區的半導體廠，廢水排入園區污水下水道，請問應該看半導體業放流水標準還是科學工業園區專用污水下水道系統標準？",
                "route": "sewer_vs_industry_query",
                "expected": ["附表九", "科學工業園區專用污水下水道系統", "不能只用半導體業附表一"],
                "expected_citations": ["附表九"],
            },
            {
                "question": "化工廠廢水中檢出苯、甲苯、二氯甲烷，是否有對應放流水限值？",
                "route": "limit_query",
                "expected": ["查無", "苯", "甲苯", "二氯甲烷", "不自行推測"],
                "forbidden": ["水污染防治法 第 1 條", "附表八"],
            },
        ]

        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"dataset{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
                )
                self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case.get("expected", []):
                    self.assertIn(keyword, payload["final_answer"])
                for keyword in case.get("forbidden", []):
                    self.assertNotIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                for keyword in case.get("expected_citations", []):
                    self.assertIn(keyword, titles)
                for keyword in case.get("forbidden_citations", []):
                    self.assertNotIn(keyword, titles)

    def test_water_regression_dataset_multi_numeric_compliance(self):
        called = []
        payload = process_query(
            query="我是一家電鍍廠，廢水檢測結果 pH 7.5、銅 2.5 mg/L、鎳 1.2 mg/L、六價鉻 0.6 mg/L，請判斷是否符合放流水標準，並列出依據。",
            conversation_id="dataset-compliance",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "graph answer", "law_search_results": [], "cited_articles": []},
        )
        self.assertNotEqual(called, [])
        self.assertEqual(payload["state"].get("route"), "numeric_compliance")
        self.assertEqual(payload["structured_judgment"]["overall_status"], "failed")
        self.assertIn("鎳", payload["final_answer"])
        self.assertIn("1.2", payload["final_answer"])
        self.assertIn("六價鉻", payload["final_answer"])
        self.assertIn("0.6", payload["final_answer"])
        self.assertIn("不符合", payload["final_answer"])
        titles = "\n".join(item["title"] for item in payload["citations"])
        self.assertIn("附表五", titles)
        self.assertNotIn("附表八", titles)

    def test_water_regression_dataset_second_batch(self):
        cases = [
            {
                "question": "光電廠放流水氨氮是 40 mg/L，是否超標？",
                "route": "numeric_compliance",
                "expected": ["不符合", "NH3-N", "40", "30", "附表二"],
                "expected_citations": ["附表二"],
            },
            {
                "question": "石化廠 COD 120 mg/L、SS 25 mg/L、pH 7.2，是否符合放流水標準？",
                "route": "numeric_compliance",
                "expected": ["不符合", "COD", "120", "SS", "符合", "pH", "附表三"],
                "expected_citations": ["附表三"],
            },
            {
                "question": "社區專用污水下水道系統的大腸桿菌群標準是多少？",
                "route": "limit_query",
                "expected": ["大腸桿菌群", "200,000", "CFU/100mL", "附表十二"],
                "expected_citations": ["附表十二"],
            },
            {
                "question": "公共污水下水道系統放流水氨氮超標，應該依哪個附表判斷？",
                "route": "scope_query",
                "expected": ["公共污水下水道系統", "附表十四"],
                "expected_citations": ["附表十四"],
            },
            {
                "question": "建築物污水處理設施是 98 年 1 月 1 日以後申請建造執照，放流水標準是否和較早申請者不同？",
                "route": "scope_query",
                "expected": ["會不同", "98 年 1 月 1 日", "97 年 12 月 31 日", "附表十五"],
                "expected_citations": ["附表十五"],
            },
            {
                "question": "請用表格整理半導體業、光電業、石化業、化工業的 pH、COD、SS、NH3-N 標準。",
                "route": "multi_industry_limit_table",
                "expected": ["半導體業", "光電業", "石化業", "化工業", "pH", "COD", "SS", "NH3-N", "查無結構化限值"],
                "expected_citations": ["附表一", "附表二", "附表三", "附表四"],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"dataset2-{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong", "law_search_results": [], "cited_articles": []},
                )
                if case["route"] != "numeric_compliance":
                    self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case["expected"]:
                    self.assertIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                for keyword in case["expected_citations"]:
                    self.assertIn(keyword, titles)

    def test_remaining_water_dataset_cases_are_structured(self):
        cases = [
            {
                "question": "發電廠放流水中的油脂、氨氮、pH 標準是多少？",
                "route": "limit_query",
                "expected": ["油脂", "10 mg/L", "NH3-N", "60 mg/L", "pH", "6.0", "9.0", "附表六"],
                "expected_citations": ["附表六"],
            },
            {
                "question": "海水淡化廠的 COD、SS、pH 放流水限值是多少？",
                "route": "limit_query",
                "expected": ["COD", "100 mg/L", "SS", "50 mg/L", "pH", "6.0", "9.0", "附表七"],
                "expected_citations": ["附表七"],
            },
            {
                "question": "建築物污水處理設施的 BOD、COD、SS 標準是多少？",
                "route": "limit_query",
                "expected": ["BOD", "COD", "SS", "附表十五"],
                "expected_citations": ["附表十五"],
            },
            {
                "question": "我們公司是金屬零件酸洗及電鍍加工，放流水標準應該看哪一個附表？",
                "route": "scope_query",
                "expected": ["金屬表面處理業", "電鍍業", "附表五"],
                "expected_citations": ["附表五"],
            },
            {
                "question": "如果我是一般食品工廠，不屬於半導體、光電、石化、化工、金屬、發電廠或海水淡化廠，應該適用哪個放流水標準？",
                "route": "scope_query",
                "expected": ["以外之事業", "附表八"],
                "expected_citations": ["附表八"],
            },
            {
                "question": "石油化學專業區專用污水下水道系統和一般石油化學業的放流水標準有什麼不同？",
                "route": "sewer_vs_industry_query",
                "expected": ["石油化學專業區專用污水下水道系統", "石油化學業", "附表十", "附表三"],
                "expected_citations": ["附表十", "附表三"],
            },
            {
                "question": "科學工業園區專用污水下水道系統的 COD、BOD、SS 限值是多少？",
                "route": "limit_query",
                "expected": ["COD", "BOD", "SS", "附表九"],
                "expected_citations": ["附表九"],
            },
            {
                "question": "其他工業區專用污水下水道系統放流水要檢測哪些常見項目？",
                "route": "item_list_query",
                "expected": ["附表十一", "COD", "SS", "pH"],
                "expected_citations": ["附表十一"],
            },
            {
                "question": "公共污水下水道系統如果每日流量大於 250 立方公尺，適用標準有不同嗎？",
                "route": "scope_query",
                "expected": ["公共污水下水道系統", "250", "附表十四"],
                "expected_citations": ["附表十四"],
            },
            {
                "question": "建築物污水處理設施和社區專用污水下水道系統的放流水標準差在哪？",
                "route": "sewer_vs_industry_query",
                "expected": ["建築物污水處理設施", "社區專用污水下水道系統", "附表十五", "附表十二"],
                "expected_citations": ["附表十五", "附表十二"],
            },
            {
                "question": "半導體廠放流水 pH 是 5.8，是否符合標準？",
                "route": "numeric_compliance",
                "expected": ["不符合", "5.8", "pH", "6.0", "9.0", "附表一"],
                "expected_citations": ["附表一"],
            },
            {
                "question": "電鍍廠六價鉻檢測值 0.8 mg/L，是否違反放流水標準？",
                "route": "numeric_compliance",
                "expected": ["不符合", "六價鉻", "0.8", "0.5", "附表五"],
                "expected_citations": ["附表五"],
            },
            {
                "question": "建築物污水處理設施 BOD 35 mg/L、COD 90 mg/L、SS 40 mg/L，是否合格？",
                "route": "numeric_compliance",
                "expected": ["BOD", "COD", "SS", "40", "附表十五"],
                "expected_citations": ["附表十五"],
            },
            {
                "question": "社區污水下水道放流水大腸桿菌群檢測值偏高，可能違反哪個標準？",
                "route": "limit_query",
                "expected": ["社區專用污水下水道系統", "大腸桿菌群", "200,000", "附表十二"],
                "expected_citations": ["附表十二"],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"remaining-water-{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong", "law_search_results": [], "cited_articles": []},
                )
                if case["route"] != "numeric_compliance":
                    self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case["expected"]:
                    self.assertIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                for keyword in case["expected_citations"]:
                    self.assertIn(keyword, titles)

    def test_anti_hallucination_guardrail_queries_do_not_call_rag(self):
        cases = [
            {
                "question": "請列出你依據的法規名稱、附表名稱和污染物限值，不要只給結論。",
                "route": "anti_reference_requirements",
                "expected": ["法規名稱", "附表名稱", "污染物限值", "單位", "不自行推測"],
            },
            {
                "question": "如果我的行業別不明確，請先問我需要補充哪些資訊，不要直接套用標準。",
                "route": "missing_industry_guardrail",
                "expected": ["行業別", "不能直接套用", "排放去向", "污染物項目", "保護區"],
            },
            {
                "question": "請區分「事業放流水標準」和「專用污水下水道系統放流水標準」。",
                "route": "effluent_vs_sewer_standard",
                "expected": ["事業放流水標準", "專用污水下水道系統放流水標準", "排放主體", "排放去向", "不能"],
            },
            {
                "question": "如果資料庫沒有某個污染物的標準，請明確說明查無資料，不要自行推測。",
                "route": "no_standard_guardrail",
                "expected": ["查無", "不自行推測", "污染物別名", "排放去向"],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"anti-hallucination-{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong", "law_search_results": [], "cited_articles": []},
                )
                self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case["expected"]:
                    self.assertIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                self.assertIn("放流水標準第2條", titles)
                self.assertNotIn("水污染防治措施及檢測申報管理辦法", titles)

    def test_water_obligation_queries_do_not_fall_back_to_empty_rag(self):
        cases = [
            {
                "question": "事業排放廢水前，需要申請水污染防治許可嗎？",
                "route": "permit_plan",
                "expected": ["通常需要", "水污染防治措施計畫", "排放許可證"],
                "expected_citations": ["水污染防治法", "許可申請審查管理辦法"],
            },
            {
                "question": "水污染防治措施計畫和排放許可證有什麼差別？",
                "route": "permit_plan",
                "expected": ["不同階段", "處理系統設計", "實際排放"],
                "expected_citations": ["水污染防治法", "許可申請審查管理辦法"],
            },
            {
                "question": "工廠新增廢水處理設備，是否需要變更水污染防治許可？",
                "route": "permit_plan",
                "expected": ["可能需要", "變更", "影響原核准內容"],
                "expected_citations": ["許可申請審查管理辦法"],
            },
            {
                "question": "廢水處理設施操作紀錄需要保存嗎？要記錄哪些內容？",
                "route": "monitoring_reporting",
                "expected": ["需要", "操作時間", "水量", "水質", "加藥", "設備", "異常"],
                "expected_citations": ["水污染防治措施及檢測申報管理辦法"],
            },
            {
                "question": "水質檢測申報通常需要包含哪些資料？",
                "route": "monitoring_reporting",
                "expected": ["採樣", "檢測項目", "檢測結果", "檢測方法", "檢測單位"],
                "expected_citations": ["水污染防治措施及檢測申報管理辦法"],
            },
            {
                "question": "事業排放廢水超過放流水標準，會依水污染防治法怎麼處理？",
                "route": "penalty_violation",
                "expected": ["罰鍰", "限期改善", "連續處罰"],
                "expected_citations": ["水污染防治法", "裁罰準則"],
            },
            {
                "question": "未取得許可就排放廢水，可能會被怎麼裁罰？",
                "route": "penalty_violation",
                "expected": ["未取得", "本身即可能", "罰鍰", "停止排放"],
                "expected_citations": ["水污染防治法", "裁罰準則"],
            },
            {
                "question": "請幫我建立一份水污染法規合規檢查清單，包含許可、操作紀錄、檢測申報、放流水標準與裁罰風險。",
                "route": "water_compliance_checklist",
                "expected": ["檢查清單", "許可文件", "操作紀錄", "檢測申報", "放流水標準", "裁罰風險"],
                "expected_citations": ["水污染防治法", "檢測申報管理辦法", "裁罰準則"],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"water-obligation-{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong", "law_search_results": [], "cited_articles": []},
                )
                self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case["expected"]:
                    self.assertIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                for keyword in case["expected_citations"]:
                    self.assertIn(keyword, titles)
                self.assertNotIn("放流水標準第2條附表八", titles)

    def test_water_obligation_remaining_detail_cases(self):
        cases = [
            {
                "question": "如果排放水量增加，原本的許可文件還能繼續使用嗎？",
                "route": "permit_plan",
                "expected": ["可能需要", "變更", "排放水量增加", "原許可文件"],
                "expected_citations": ["許可申請審查管理辦法"],
            },
            {
                "question": "如果沒有依規定做檢測申報，可能違反哪些水污染防治規定？",
                "route": "monitoring_reporting",
                "expected": ["未定期檢測", "不實申報", "罰鍰", "限期改善"],
                "expected_citations": ["檢測申報管理辦法"],
            },
            {
                "question": "繞流排放和一般水質超標的違規嚴重程度有差嗎？",
                "route": "penalty_violation",
                "expected": ["繞流", "更嚴重", "罰鍰", "停工停業"],
                "expected_citations": ["水污染防治法", "裁罰準則"],
            },
            {
                "question": "如果廢水處理設施沒有正常操作，但檢測值尚未超標，是否仍可能違規？",
                "route": "penalty_violation",
                "expected": ["仍可能違規", "正常操作", "未超標", "管理義務"],
                "expected_citations": ["水污染防治法", "裁罰準則"],
            },
            {
                "question": "違反水污染防治法罰鍰額度裁罰準則通常會考量哪些因素？",
                "route": "penalty_violation",
                "expected": ["違規類型", "超標程度", "違規次數", "改善配合"],
                "expected_citations": ["水污染防治法", "裁罰準則"],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=case["question"]):
                called = []
                payload = process_query(
                    query=case["question"],
                    conversation_id=f"water-obligation-detail-{index}",
                    category="auto",
                    run_graph=lambda query: called.append(query) or {"final_answer": "wrong", "law_search_results": [], "cited_articles": []},
                )
                self.assertEqual(called, [])
                self.assertEqual(payload["state"].get("route"), case["route"])
                for keyword in case["expected"]:
                    self.assertIn(keyword, payload["final_answer"])
                titles = "\n".join(item["title"] for item in payload["citations"])
                for keyword in case["expected_citations"]:
                    self.assertIn(keyword, titles)
                self.assertNotIn("放流水標準第2條附表", titles)

    def test_waste_solvent_obligation_query_does_not_request_numeric_data(self):
        called = []
        payload = process_query(
            query="\u6211\u5011\u5de5\u5ee0\u53ea\u662f\u5076\u723e\u7522\u751f\u5c11\u91cf\u5ee2\u6eb6\u5291\uff0c\u4e5f\u9700\u8981\u59d4\u8a17\u5408\u6cd5\u6e05\u9664\u8655\u7406\u6a5f\u69cb\u55ce\uff1f",
            conversation_id="waste1",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertFalse(payload["waiting_for_data_input"])
        self.assertIn("\u901a\u5e38\u4ecd\u9700", payload["final_answer"])
        self.assertIn("\u59d4\u8a17\u5408\u6cd5\u6e05\u9664\u8655\u7406\u6a5f\u69cb", payload["final_answer"])
        self.assertIn("\u5c11\u91cf", payload["final_answer"])
        self.assertIn("\u4e0d\u4ee3\u8868", payload["final_answer"])
        self.assertNotIn("\u8acb\u63d0\u4f9b", payload["final_answer"])
        titles = "\n".join(item["title"] for item in payload["citations"])
        self.assertIn("\u5ee2\u68c4\u7269\u6e05\u7406\u6cd5", titles)
        self.assertNotIn("\u653e\u6d41\u6c34", titles)
        self.assertNotIn("\u6c34\u6c61\u67d3\u9632\u6cbb", titles)

    def test_waste_solvent_hazardous_classification_requests_sds_details(self):
        called = []
        payload = process_query(
            query="\u9019\u6279\u5ee2\u6eb6\u5291\u662f\u4e0d\u662f\u6709\u5bb3\u4e8b\u696d\u5ee2\u68c4\u7269\uff1f",
            conversation_id="waste2",
            category="auto",
            run_graph=lambda query: called.append(query) or {"final_answer": "wrong"},
        )
        self.assertEqual(called, [])
        self.assertIn("SDS", payload["final_answer"])
        self.assertIn("\u6210\u5206", payload["final_answer"])
        self.assertIn("\u9583\u706b\u9ede", payload["final_answer"])
        self.assertIn("\u5ee2\u68c4\u7269\u4ee3\u78bc", payload["final_answer"])

    def test_new_waste_topic_does_not_inherit_previous_effluent_memory(self):
        called = []

        def fake_graph(query):
            called.append(query)
            return {"final_answer": "graph answer", "law_search_results": [], "cited_articles": []}

        process_query(
            query="\u77f3\u6cb9\u5316\u5b78\u696d COD 120 mg/L \u662f\u5426\u5408\u898f\uff1f",
            conversation_id="mixed1",
            category="effluent_standard",
            run_graph=fake_graph,
        )
        second = process_query(
            query="\u6211\u5011\u5de5\u5ee0\u53ea\u662f\u5076\u723e\u7522\u751f\u5c11\u91cf\u5ee2\u6eb6\u5291\uff0c\u4e5f\u9700\u8981\u59d4\u8a17\u5408\u6cd5\u6e05\u9664\u8655\u7406\u6a5f\u69cb\u55ce\uff1f",
            conversation_id="mixed1",
            category="auto",
            run_graph=fake_graph,
        )
        self.assertEqual(len(called), 1)
        self.assertIn("\u59d4\u8a17\u5408\u6cd5\u6e05\u9664\u8655\u7406\u6a5f\u69cb", second["final_answer"])
        self.assertNotIn("\u77f3\u6cb9\u5316\u5b78\u696d", second["final_answer"])
        self.assertNotIn("COD", second["final_answer"])
        self.assertNotIn("\u653e\u6d41\u6c34", "\n".join(item["title"] for item in second["citations"]))

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

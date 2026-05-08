import unittest

from pollutant_catalog import infer_industry_hint, parse_measurements
from standards import evaluate_records


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


if __name__ == "__main__":
    unittest.main()

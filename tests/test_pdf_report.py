import unittest
from utils.pdf_report import FinTwinPDFReport, _clean_text_for_pdf

class TestPDFReport(unittest.TestCase):
    def test_clean_text_with_font_awesome(self):
        text = '<i class="fa-solid fa-arrow-trend-down" style="color:#EF4444; margin-right:4px;"></i> <b>Savings Rate</b> \u2014 your savings rate of 0.0% reduced your score by <b>+12.5 pts</b>.'
        cleaned = _clean_text_for_pdf(text)
        self.assertNotIn('<i class=', cleaned)
        self.assertNotIn('</i>', cleaned)
        self.assertIn('<b>Savings Rate</b>', cleaned)
        self.assertIn('(-) ', cleaned)

    def test_pdf_build_with_icon_paragraphs(self):
        rep = FinTwinPDFReport(report_title="Test Report", user_id="test_user")
        rep.add_paragraph('<i class="fa-solid fa-arrow-trend-up" style="color:#10B981;"></i> <b>Income Growth</b> increased score.')
        rep.add_paragraph('<i class="fa-solid fa-arrow-trend-down" style="color:#EF4444;"></i> <b>High EMI</b> decreased score.')
        rep.add_paragraph('<span style="color:#EF4444;">Critical alert</span> text.')
        pdf_bytes = rep.build()
        self.assertTrue(len(pdf_bytes) > 1000)

    def test_new_report_design_components(self):
        import pandas as pd
        rep = FinTwinPDFReport(
            report_title="Comprehensive Wealth Report",
            user_id="USR-TEST-001",
            report_id="TEST-REP-01",
            subtitle="Test Subtitle for Report Validation",
        )
        rep.add_cover_page(user_name="Test User", report_type="Comprehensive Wealth Report")
        rep.add_table_of_contents(["Executive Summary", "KPI Metrics", "Detailed Tables"])
        rep.add_section_divider("Executive Summary")
        rep.add_executive_summary("This is an executive summary of financial health.", {
            "Net Worth": "Rs. 25,00,000",
            "Health Score": "82 / 100",
        })
        rep.add_kpi_summary_row([
            {"label": "Monthly Surplus", "value": "Rs. 45,000", "status": "positive"},
            {"label": "Debt Ratio", "value": "15%", "status": "neutral"},
            {"label": "Risk Factor", "value": "Low", "status": "positive"},
        ])
        rep.add_callout("All systems operational. Savings rate exceeds target.", style="success", title="Savings Milestone")
        rep.add_callout("High discretionary spending detected in dining.", style="warning", title="Budget Alert")
        rep.add_callout("Tax filing deadline approaching.", style="info", title="Tax Notice")
        rep.add_callout("Emergency fund below 3 months.", style="danger", title="Emergency Fund Alert")
        rep.add_section_divider("Detailed Tables")
        df = pd.DataFrame({"Metric": ["Income", "Expense", "Savings"], "Value": [100000, 55000, 45000]})
        rep.add_dataframe_table(df, title="Cash Flow Table")
        rep.add_key_value_grid({"Annual Income": "Rs. 12,00,000", "Tax Regime": "New Regime"})
        pdf_bytes = rep.build()
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 5000)

if __name__ == '__main__':
    unittest.main()


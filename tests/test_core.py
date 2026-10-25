from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from zentromart.agent import SupportAgent
from zentromart.knowledge import KnowledgeBase
from zentromart.tools import SupportTools


class FakeClient:
    def __init__(self, response: str = "Please see the current policy. [RET-2]"):
        self.response = response
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        return {"content": self.response}


class CoreTests(unittest.TestCase):
    def test_parser_excludes_superseded_and_search_returns_section_ids(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "current.md").write_text("---\ndoc_type: policy\nstatus: current\neffective_from: 2026-01-01\n---\n## [RET-2] Return windows\nReturn window is seven calendar days.", encoding="utf-8")
            (p / "old.md").write_text("---\ndoc_type: policy\nstatus: superseded\neffective_from: 2025-01-01\n---\n## [OLD-2] Return window\nReturn window is thirty days.", encoding="utf-8")
            kb = KnowledgeBase.from_directory(p)
            self.assertEqual([x.source_id for x, _ in kb.search("return window days")], ["RET-2"])

    def test_tool_does_not_reveal_order_before_verification(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            def write(name, rows):
                with (p / name).open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            write("customers.csv", [{"customer_id":"C1", "phone":"+91 5000012345"}])
            write("orders.csv", [{"order_id":"12345678", "customer_id":"C1", "delivery_pincode":"560001", "status":"delivered", "total_inr":"100"}])
            write("products.csv", [{"sku":"A", "title":"Test"}])
            write("order_items.csv", [{"order_id":"12345678", "sku":"A", "qty":"1", "unit_price_inr":"100"}])
            tools = SupportTools(p)
            pending = tools.lookup_order("12345678")
            self.assertNotIn("status", pending)
            self.assertNotIn("items", pending)
            self.assertFalse(tools.lookup_order("12345678", "0000").get("verified"))
            verified = tools.lookup_order("12345678", "2345")
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["status"], "delivered")

    def test_agent_attaches_retrieved_section_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); kbdir = root / "kb"; datadir = root / "data"
            kbdir.mkdir(); datadir.mkdir()
            (kbdir / "policy.md").write_text("---\ndoc_type: policy\nstatus: current\n---\n## [RET-2] Return window\nReturn window is seven days.", encoding="utf-8")
            for name, fields in {"customers.csv":["customer_id"], "orders.csv":["order_id"], "products.csv":["sku"], "order_items.csv":["order_id"]}.items():
                (datadir / name).write_text(",".join(fields)+"\n", encoding="utf-8")
            client = FakeClient()
            agent = SupportAgent(KnowledgeBase.from_directory(kbdir), datadir, client)
            result = agent.respond("What is the return window?")
            self.assertEqual(result.sources, ["RET-2"])
            self.assertTrue(client.calls)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class SupportTools:
    """Synthetic-data tools. Identity checks gate all private order facts."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.customers = self._rows("customers.csv", "customer_id")
        self.orders = self._rows("orders.csv", "order_id")
        self.items = self._list("order_items.csv")
        self.products = self._rows("products.csv", "sku")
        self._verified: dict[str, dict[str, Any]] = {}
        self.ticket_path = data_dir.parent / "tools" / "tickets_created.csv"

    def _list(self, name: str) -> list[dict[str, str]]:
        with (self.data_dir / name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _rows(self, name: str, key: str) -> dict[str, dict[str, str]]:
        return {row[key]: row for row in self._list(name)}

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict[str, Any]:
        order_id = "".join(c for c in str(order_id) if c.isdigit())
        order = self.orders.get(order_id)
        if verification_value is None:
            return {"verification_required": True,
                    "message": "Ask for the last four digits of the registered phone or the six-digit delivery pincode."}
        if not order:
            return {"verified": False, "verification_required": True, "message": "Verification failed or order not found."}
        customer = self.customers.get(order.get("customer_id", ""), {})
        candidate = "".join(c for c in str(verification_value) if c.isdigit())
        phone = "".join(c for c in customer.get("phone", "") if c.isdigit())
        valid = (len(candidate) == 4 and phone.endswith(candidate)) or (
            len(candidate) == 6 and candidate == "".join(c for c in order.get("delivery_pincode", "") if c.isdigit()))
        if not valid:
            return {"verified": False, "verification_required": True, "message": "Verification failed or order not found."}
        self._verified[order_id] = order
        fields = ("order_id", "status", "promised_date", "payment_method", "delivery_type", "delivered_on",
                  "return_status", "return_requested_on", "return_refund_initiated_on", "return_refund_completed_on",
                  "damage_status", "damage_issue", "total_inr", "timeline")
        safe = {k: order.get(k, "") for k in fields}
        safe["items"] = [{"sku": x.get("sku", ""), "title": self.products.get(x.get("sku", ""), {}).get("title", ""),
                          "qty": x.get("qty", ""), "unit_price_inr": x.get("unit_price_inr", "")}
                         for x in self.items if x.get("order_id") == order_id]
        safe["verified"] = True
        return safe

    def create_ticket(self, summary: str, order_id: str = "", reason: str = "general") -> dict[str, Any]:
        normalized = "".join(c for c in str(order_id) if c.isdigit())
        if normalized and normalized not in self._verified:
            return {"created": False, "error": "Order must be verified before an order-specific ticket is created."}
        if len(summary.strip()) < 20:
            return {"created": False, "error": "Include the issue, checks made, and escalation reason in the summary."}
        self.ticket_path.parent.mkdir(parents=True, exist_ok=True)
        ticket_id = "TKT" + uuid4().hex
        row = {"ticket_id": ticket_id, "created_at": datetime.now().isoformat(timespec="seconds"),
               "order_id": normalized, "reason": reason, "summary": summary[:500]}
        fresh = not self.ticket_path.exists()
        with self.ticket_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        return {"created": True, "ticket_id": ticket_id}


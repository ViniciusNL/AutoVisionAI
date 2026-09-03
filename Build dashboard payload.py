"""Aggregates data/inspections_raw.json (real engine output on synthetic
photos) into the compact JSON payload consumed by dashboard.html."""

import json
import collections
import statistics as st

IN_PATH = "/home/claude/autovision/data/inspections_raw.json"
OUT_PATH = "/home/claude/autovision/data/dashboard_data.json"

DECISION_META = {
    "aprovado_automatico": {"label": "Aprovado automaticamente", "color": "#22C58B", "icon": "check"},
    "revisao_manual": {"label": "Revisão manual", "color": "#F5A623", "icon": "flag"},
    "fraude_suspeita": {"label": "Fraude suspeita", "color": "#F0475B", "icon": "alert"},
}

SEVERITY_META = {
    "leve": {"label": "Leve", "color": "#22C58B"},
    "moderado": {"label": "Moderado", "color": "#F5A623"},
    "grave": {"label": "Grave", "color": "#F0475B"},
    "critico": {"label": "Crítico", "color": "#B91C3C"},
}

CATEGORY_META = {
    "amassado": {"label": "Amassados", "color": "#3B82F6"},
    "arranhao": {"label": "Arranhões", "color": "#22C58B"},
    "vidro": {"label": "Vidro / Para-brisa", "color": "#F5A623"},
    "farol_lanterna": {"label": "Farol / Lanterna", "color": "#8B5CF6"},
    "estrutural": {"label": "Estrutural / Chassi", "color": "#F0475B"},
    "pneu_roda": {"label": "Pneu / Roda", "color": "#2DD4BF"},
}


def build():
    data = json.load(open(IN_PATH, encoding="utf-8"))
    n = len(data)

    total_cost = sum(d["estimated_cost"] for d in data)
    avg_ticket = total_cost / n
    n_auto = sum(1 for d in data if d["decision"] == "aprovado_automatico")
    n_manual = sum(1 for d in data if d["decision"] == "revisao_manual")
    n_fraud = sum(1 for d in data if d["decision"] == "fraude_suspeita")
    auto_rate = n_auto / n * 100
    fraud_rate = n_fraud / n * 100

    # -- yearly trend -------------------------------------------------
    by_year = collections.defaultdict(list)
    for d in data:
        by_year[d["date"][:4]].append(d)
    years = sorted(by_year.keys())
    yearly_trend = []
    for y in years:
        recs = by_year[y]
        yearly_trend.append({
            "year": y,
            "count": len(recs),
            "cost": round(sum(r["estimated_cost"] for r in recs), 2),
            "status": "current" if y == "2026" else ("record" if y == "2025" else "closed"),
        })

    # -- damage category mix (by total estimated cost) ----------------
    cat_cost = collections.Counter()
    cat_count = collections.Counter()
    for d in data:
        for r in d["damage_regions"]:
            cat_cost[r["category"]] += 0  # placeholder, replaced below
    # cost isn't stored per-region in the raw record, so approximate the
    # split proportionally to how often each category appears weighted
    # by its severity, which is how estimate_cost() derives cost too.
    cat_weight = collections.Counter()
    for d in data:
        for r in d["damage_regions"]:
            base = {"amassado": 850, "arranhao": 320, "vidro": 1100,
                    "farol_lanterna": 480, "estrutural": 4200, "pneu_roda": 560}[r["category"]]
            mult = 0.4 + (r["severity"] / 100) * 1.6
            cat_weight[r["category"]] += base * mult
            cat_count[r["category"]] += 1

    total_weight = sum(cat_weight.values()) or 1
    damage_mix = []
    for cat, meta in CATEGORY_META.items():
        w = cat_weight.get(cat, 0)
        damage_mix.append({
            "key": cat,
            "label": meta["label"],
            "color": meta["color"],
            "cost": round(w, 2),
            "count": cat_count.get(cat, 0),
            "pct": round(w / total_weight * 100, 1),
        })
    damage_mix.sort(key=lambda x: x["cost"], reverse=True)

    # -- recent inspections table (full record, so any row can open the
    #    detail/laudo view, not only the curated highlight cases) -------
    recent = []
    for d in data[:26]:
        primary_cat = None
        if d["damage_regions"]:
            primary_cat = max(d["damage_regions"], key=lambda r: r["severity"])["category"]
        recent.append({
            "vehicle_id": d["vehicle_id"],
            "plate_masked": d["plate_masked"],
            "make": d["make"],
            "model": d["model"],
            "date": d["date"],
            "insurer": d["insurer"],
            "state": d["state"],
            "n_images": d["n_images"],
            "severity_score": d["severity_score"],
            "severity_band": d["severity_band"],
            "fraud_score": d["fraud_score"],
            "estimated_cost": d["estimated_cost"],
            "decision": d["decision"],
            "decision_reason": d["decision_reason"],
            "primary_category": primary_cat,
            "damage_regions": d["damage_regions"],
            "tamper_signals": d["tamper_signals"],
            "quality": d["quality"],
        })

    # -- highlighted cases (used to pre-open a representative example
    #    of each outcome the first time the viewer opens the laudo) ----
    def pick(decision, severity_band=None):
        pool = [d for d in recent if d["decision"] == decision
                and (severity_band is None or d["severity_band"] == severity_band)]
        pool.sort(key=lambda d: len(d["damage_regions"]) + len(d["tamper_signals"]), reverse=True)
        return pool[0]["vehicle_id"] if pool else None

    highlight_ids = [vid for vid in [
        pick("aprovado_automatico", "leve"),
        pick("revisao_manual", "grave"),
        pick("fraude_suspeita", None),
    ] if vid]

    payload = {
        "generated_at": data[0]["date"] if data else None,
        "kpis": {
            "total_inspections": n,
            "total_estimated_cost": round(total_cost, 2),
            "avg_ticket": round(avg_ticket, 2),
            "auto_approval_rate": round(auto_rate, 1),
            "n_auto": n_auto,
            "n_manual": n_manual,
            "fraud_rate": round(fraud_rate, 1),
            "n_fraud": n_fraud,
        },
        "yearly_trend": yearly_trend,
        "damage_mix": damage_mix,
        "recent_inspections": recent,
        "highlight_ids": highlight_ids,
        "decision_meta": DECISION_META,
        "severity_meta": SEVERITY_META,
        "category_meta": CATEGORY_META,
    }
    return payload


if __name__ == "__main__":
    payload = build()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("KPIs:", payload["kpis"])
    print("Yearly trend:", payload["yearly_trend"])
    print("Damage mix:", [(m["label"], m["pct"]) for m in payload["damage_mix"]])
    print("Wrote", OUT_PATH)
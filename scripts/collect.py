"""Run the data collection.

  python scripts/collect.py daily      # every day: new values from every source
  python scripts/collect.py backfill   # once: full history since 2019 (EC since Aug 2023)

Writes data/clean/*.csv and data/status.json. Exits with code 1 if a core source failed,
so GitHub Actions marks the run red and emails the repository owner.
"""
from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrows.common import log, now_utc, write_status  # noqa: E402
from narrows import weekly  # noqa: E402
from narrows.sources import context, drewry, ec_futures, gdelt, markets, portwatch  # noqa: E402

RECENT = (date.today() - timedelta(days=21)).isoformat()

JOBS = {
    "daily": [
        ("drewry", True, drewry.fetch),
        ("ec_futures", True, ec_futures.fetch),
        ("portwatch_straits", True, lambda: portwatch.chokepoints(start=RECENT)),
        ("portwatch_ports", False, lambda: portwatch.ports(start=RECENT)),
        ("markets", True, lambda: markets.fetch(start=RECENT)),
        ("news_articles", False, gdelt.articles),
        ("harpex", False, context.harpex),
        ("bunker", False, context.bunker),
        ("drewry_weekly", False, weekly.build),
    ],
    "recent_news": [
        ("news_recent_15d", True, gdelt.articles_range),
    ],
    "backfill": [
        ("drewry", True, drewry.fetch),
        ("drewry_press", False, drewry.backfill_from_press),
        ("ec_futures", True, ec_futures.fetch),
        ("portwatch_straits", True, portwatch.chokepoints),
        ("portwatch_ports", False, portwatch.ports),
        ("markets", True, markets.fetch),
        ("news_volume", False, gdelt.history),
        ("harpex", False, context.harpex),
        ("bunker", False, context.bunker),
        ("drewry_weekly", False, weekly.build),
    ],
}


def main(mode: str) -> int:
    results, core_failed = {}, []
    for name, core, fn in JOBS[mode]:
        try:
            df = fn()
            rows = 0 if df is None else len(df)
            results[name] = {"ok": True, "rows": rows, "at": now_utc(), "core": core}
        except Exception as e:  # keep going, report at the end
            log.error("%s failed: %s", name, e)
            traceback.print_exc()
            results[name] = {"ok": False, "error": str(e)[:300], "at": now_utc(), "core": core}
            if core:
                core_failed.append(name)
    write_status(results)
    print("\nSUMMARY")
    for k, v in results.items():
        print(f"  {'OK  ' if v['ok'] else 'FAIL'} {k:<18} {v.get('rows', v.get('error', ''))}")
    if core_failed:
        print("Core sources failed:", ", ".join(core_failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "daily"))

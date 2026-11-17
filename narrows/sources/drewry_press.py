"""Rebuild Drewry WCI weeks that the open chart does not cover, from press reprints of Drewry's weekly release.

Drewry sells its full history; the open chart (akshare) never carried Shanghai -> New York, and its backhaul and
transatlantic series stopped on 17 Sep 2026. Every Thursday Drewry also publishes a text release ("Freight rates
from Shanghai to New York decreased 4% or $184 to $4,710 per feu"), reprinted by trade sites. This module:

  1. crawls those reprints (WordPress REST API of two trade sites, plus Drewry's own page),
  2. parses each release: assessment date, composite, every route quoted with its value, and the previous week's
     value when the release gives the dollar change ("or $184"),
  3. checks every release against the values we already hold from Drewry's chart (Rotterdam, Genoa, Los Angeles
     and, until Sep 2026, the other routes). A release is kept only if every value we can check matches to $1;
     a release with nothing to check is kept only where a second accepted source gives the same value,
  4. writes the accepted values with their source, and a report of the checks.

Release texts are cached in data/raw/drewry_press_posts.jsonl, so the daily run only reads the newest search page.
Outputs: data/raw/drewry_press_parsed.csv (every parsed value, accepted or not, with its source),
data/clean/drewry_press.csv (accepted values), data/clean/drewry_press_report.json (the checks).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import pandas as pd

from ..common import CLEAN, HISTORY_START, RAW, get, log, now_utc

SOURCES = {
    "cyprusshippingnews": "https://cyprusshippingnews.com/wp-json/wp/v2/posts",
    "hellenicshippingnews": "https://www.hellenicshippingnews.com/wp-json/wp/v2/posts",
}
SEARCH_TERMS = ["World Container Index", "Drewry WCI"]

CITY = r"(Shanghai|Rotterdam|Genoa|Los Angeles|New York)"
PAIR_RE = re.compile(CITY + r"\s*(?:to|–|—|-)\s*" + CITY, re.I)
PAIRS = {
    ("shanghai", "rotterdam"): "shanghai_rotterdam",
    ("shanghai", "genoa"): "shanghai_genoa",
    ("shanghai", "los angeles"): "shanghai_los_angeles",
    ("shanghai", "new york"): "shanghai_new_york",
    ("rotterdam", "shanghai"): "rotterdam_shanghai",
    ("los angeles", "shanghai"): "los_angeles_shanghai",
    ("new york", "rotterdam"): "new_york_rotterdam",
    ("rotterdam", "new york"): "rotterdam_new_york",
}
UP = r"\b(?:increas|rose\b|rise|rising|jump|surg|climb|grew\b|grow|gain|add(?:ed|ing|s)?\b|edg\w* up|inch\w* up|up\b|advanc|soar|spik|rebound|firm)"
DOWN = r"\b(?:decreas|fell\b|fall|drop|declin|dip|slip|slid|eas(?:ed|es|ing)\b|lost\b|los(?:es|ing)\b|down\b|plung|tumbl|retreat|sank\b|sink|soften|weaken|shed\b|reduc)"
AMOUNT = r"\$\s?([\d,]{3,7})"
DELTA = r"\bor\s*\$\s?([\d,]{1,7})"  # "rose 3% or $25 to $785": the dollar change
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'")
    text = re.sub(r"US\$", "$", text)
    return re.sub(r"\s+", " ", text)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[a-z0-9%)\]])\.\s+(?=[A-Z])", text)


def _direction(seg: str) -> int:
    seg = seg.lower()
    m_up, m_dn = re.search(UP, seg), re.search(DOWN, seg)
    if m_up and not m_dn:
        return 1
    if m_dn and not m_up:
        return -1
    if m_up and m_dn:  # the verb nearest the start of the clause wins
        return 1 if m_up.start() < m_dn.start() else -1
    return 0


def parse_date(text: str) -> str | None:
    m = re.search(r"assessment for (?:Thursday,?\s*)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if not m:
        m2 = re.search(r"assessment for (?:Thursday,?\s*)?([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text)
        if not m2:
            return None
        day, mon, year = m2.group(2), m2.group(1), m2.group(3)
    else:
        day, mon, year = m.group(1), m.group(2), m.group(3)
    if mon[:3].lower() not in MONTHS:
        return None
    d = pd.Timestamp(year=int(year), month=MONTHS.index(mon[:3].lower()) + 1, day=int(day))
    return d.strftime("%Y-%m-%d") if d.dayofweek == 3 else None  # Drewry assesses on Thursdays


def parse_release(text: str) -> list[dict]:
    """Every value quoted in one release: [{route, value, kind}] with kind 'direct' or 'prev_week'."""
    text = _clean(text)
    out = []
    for sent in _sentences(text):
        mentions = [(m, PAIRS.get((m.group(1).lower(), m.group(2).lower()))) for m in PAIR_RE.finditer(sent)]
        mentions = [(m, r) for m, r in mentions if r]
        if not mentions:
            if re.search(r"composite|World Container Index|\bWCI\b", sent, re.I):
                m = re.search(r"(?:composite(?: index)?|World Container Index|\bWCI\b)[^$]{0,90}?(?:\bto|\bat)\s*" + AMOUNT, sent, re.I)
                if m:
                    out.append({"route": "composite", "value": _num(m.group(1)), "kind": "direct"})
                    d = re.search(DELTA, sent[m.start():m.end()], re.I)
                    sign = _direction(sent[m.start():m.end()])
                    if d and sign:
                        out.append({"route": "composite", "value": _num(m.group(1)) - sign * _num(d.group(1)), "kind": "prev_week"})
            continue
        pending = []
        for i, (m, route) in enumerate(mentions):
            end = mentions[i + 1][0].start() if i + 1 < len(mentions) else len(sent)
            seg = sent[m.end():end]
            deltas = re.findall(DELTA, seg, re.I)
            seg_v = re.sub(DELTA, " ", seg, flags=re.I)
            values = [_num(v) for v in re.findall(r"(?:\bto|\bat|\band|,)\s*" + AMOUNT, seg_v, re.I)]
            pending.append(route)
            if not values:
                continue
            if len(values) == len(pending):
                for r, v in zip(pending, values):
                    out.append({"route": r, "value": v, "kind": "direct"})
                if len(pending) == 1 and len(deltas) == 1:
                    sign = _direction(seg)
                    if sign:
                        out.append({"route": route, "value": values[0] - sign * _num(deltas[0]), "kind": "prev_week"})
            pending = []
    # one value per (route, kind); a release that quotes two different values for one route is ambiguous
    by = defaultdict(set)
    for o in out:
        by[(o["route"], o["kind"])].add(o["value"])
    return [{"route": r, "value": next(iter(v)), "kind": k} for (r, k), v in by.items() if len(v) == 1]


def _strip_html(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml").get_text(" ")


POSTS = RAW / "drewry_press_posts.jsonl"


def load_posts() -> list[dict]:
    if not POSTS.exists():
        return []
    return [json.loads(line) for line in POSTS.read_text().splitlines() if line.strip()]


def crawl(max_pages: int = 60) -> list[dict]:
    """Posts that look like a WCI release, from each source, added to the cache: [{source, url, posted, title, text}]."""
    cached = load_posts()
    posts, seen = [], {p["url"] for p in cached}
    for name, api in SOURCES.items():
        n0 = len(posts)
        for term in SEARCH_TERMS:
            for page in range(1, max_pages + 1):
                params = {"search": term, "per_page": 100, "page": page, "after": HISTORY_START + "T00:00:00",
                          "_fields": "date,link,title,content"}
                try:
                    items = get(api, params=params, retries=2).json()
                except Exception as e:
                    log.info("%s '%s' stopped at page %d: %s", name, term, page, e)
                    break
                if not isinstance(items, list) or not items:
                    break
                for it in items:
                    title = _strip_html(it.get("title", {}).get("rendered", ""))
                    if it["link"] in seen or not re.search(r"Container Index|WCI", title, re.I):
                        continue
                    seen.add(it["link"])
                    posts.append({"source": name, "url": it["link"], "posted": it["date"][:10], "title": title,
                                  "text": _strip_html(it.get("content", {}).get("rendered", ""))})
                if len(items) < 100:
                    break
        log.info("Drewry press: %s gave %d new release posts", name, len(posts) - n0)
    all_posts = cached + posts
    if posts:
        POSTS.parent.mkdir(parents=True, exist_ok=True)
        POSTS.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in sorted(all_posts, key=lambda p: p["posted"])))
    return all_posts


def build(posts: list[dict] | None = None, known: pd.DataFrame | None = None) -> pd.DataFrame:
    posts = crawl() if posts is None else posts
    if known is None:
        wci = pd.read_csv(CLEAN / "drewry_wci.csv", dtype={"usd_per_40ft": float})
        known = wci[wci["source"].isin(["drewry_infogram", "drewry_page"])]  # values straight from Drewry
    ref = {(d, r): v for d, r, v in zip(known["date"], known["route"], known["usd_per_40ft"])}

    rows = []
    for p in posts:
        d = parse_date(p["text"])
        if not d or d < HISTORY_START:
            continue
        prev = (pd.Timestamp(d) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        for v in parse_release(p["text"]):
            week = d if v["kind"] == "direct" else prev
            rows.append({"week": week, "release": d, "route": v["route"], "value": v["value"], "kind": v["kind"],
                         "source": p["source"], "url": p["url"], "known": ref.get((week, v["route"]))})
    parsed = pd.DataFrame(rows, columns=["week", "release", "route", "value", "kind", "source", "url", "known"])
    if parsed.empty:
        raise RuntimeError("Drewry press: no release could be parsed")
    parsed["checked"] = parsed["known"].notna()
    parsed["match"] = parsed["checked"] & ((parsed["value"] - parsed["known"]).abs() <= 1)

    # a release is accepted if it has at least one checkable value and every checkable value matches
    per = parsed.groupby("url").agg(n_checked=("checked", "sum"), n_match=("match", "sum"))
    per["status"] = "unverified"
    per.loc[(per["n_checked"] > 0) & (per["n_match"] == per["n_checked"]), "status"] = "accepted"
    per.loc[(per["n_checked"] > 0) & (per["n_match"] < per["n_checked"]), "status"] = "rejected"
    parsed = parsed.merge(per["status"], left_on="url", right_index=True)

    # candidate values for weeks we do not already hold
    cand = parsed[(~parsed["checked"]) & (parsed["status"] != "rejected")]
    accepted_rows, conflicts = [], 0
    for (week, route), g in cand.groupby(["week", "route"]):
        acc = g[g["status"] == "accepted"]
        values = set(acc["value"])
        n_sources = acc["url"].nunique()
        if not values:  # only unverified releases: need two distinct releases agreeing
            if g["url"].nunique() >= 2 and g["value"].nunique() == 1:
                values, n_sources = set(g["value"]), g["url"].nunique()
            else:
                continue
        if len(values) > 1:
            conflicts += 1
            continue
        best = (acc if not acc.empty else g).sort_values("kind").iloc[0]  # 'direct' before 'prev_week'
        accepted_rows.append({"date": week, "route": route, "usd_per_40ft": int(next(iter(values))),
                              "source": "press:" + best["url"], "kind": best["kind"], "n_sources": int(n_sources)})
    clean = pd.DataFrame(accepted_rows, columns=["date", "route", "usd_per_40ft", "source", "kind", "n_sources"])
    clean = clean[clean["date"] >= HISTORY_START]

    chk = parsed[parsed["checked"]]
    report = {
        "built_at": now_utc(),
        "release_posts": len(posts),
        "releases_parsed": int(parsed["url"].nunique()),
        "releases": per["status"].value_counts().to_dict(),
        "values_checked": int(len(chk)),
        "values_matching": int(chk["match"].sum()),
        "match_rate": round(float(chk["match"].mean()), 4) if len(chk) else None,
        "match_rate_prev_week": round(float(chk[chk["kind"] == "prev_week"]["match"].mean()), 4) if (chk["kind"] == "prev_week").any() else None,
        "conflicting_weeks_dropped": conflicts,
        "rebuilt_values": clean.groupby("route").size().to_dict(),
        "rebuilt_range": clean.groupby("route")["date"].agg(["min", "max"]).to_dict("index"),
        "mismatches_sample": chk[~chk["match"]].head(20)[["week", "route", "value", "known", "kind", "url"]].to_dict("records"),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(RAW / "drewry_press_parsed.csv", index=False)
    clean.to_csv(CLEAN / "drewry_press.csv", index=False)
    (CLEAN / "drewry_press_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("Drewry press: %s", {k: report[k] for k in ("releases", "values_checked", "match_rate", "rebuilt_values")})
    return clean


def merge_into_wci() -> pd.DataFrame:
    """Add the accepted press values to drewry_wci.csv where Drewry's own chart or page has nothing.

    Older press rows that were never checked are removed first.
    """
    out = CLEAN / "drewry_wci.csv"
    wci = pd.read_csv(out, dtype=str)
    wci = wci[~wci["source"].str.startswith("press:")]
    press = pd.read_csv(CLEAN / "drewry_press.csv", dtype=str)[["date", "route", "usd_per_40ft", "source"]]
    have = set(zip(wci["date"], wci["route"]))
    press = press[[(d, r) not in have for d, r in zip(press["date"], press["route"])]]
    merged = pd.concat([wci, press], ignore_index=True).sort_values(["date", "route"])
    merged.to_csv(out, index=False)
    log.info("Drewry press: %d values added to drewry_wci.csv", len(press))
    return press


def run(max_pages: int = 60) -> pd.DataFrame:
    """Full crawl (backfill) by default; the daily job passes max_pages=1 to pick up the latest releases."""
    build(crawl(max_pages))
    return merge_into_wci()

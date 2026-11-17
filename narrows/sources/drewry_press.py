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
    """Sentences, and the bullet points Drewry has used since 2026."""
    return [x for x in re.split(r"(?<=[a-z0-9%)\]])\.\s+(?=[A-Z])|\s*[•▪]\s*|;\s+", text) if x.strip()]


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


PCT = r"(\d+(?:\.\d+)?)\s*%"
# 2026 wording puts the value before the route: "fell 5% to $3,835 per 40ft container from Shanghai to Genoa"
VALUE_FIRST_RE = re.compile(
    r"(?:" + PCT + r"\s*(?:or\s*\$\s?[\d,]{1,7}\s*)?)?(?:\bto|\bat)\s*\$\s?([\d,]{3,7})\s*per\s*(?:40\s?ft|feu)"
    r"(?:\s*(?:container|box))?\s*from\s*" + CITY + r"\s*(?:to|–|—|-)\s*" + CITY, re.I)


def _pcts(seg: str) -> list[float]:
    return [float(x) for x in re.findall(PCT, re.sub(DELTA, " ", seg, flags=re.I))]


def parse_release(text: str) -> list[dict]:
    """Every value quoted in one release: [{route, value, kind, pct}].

    kind is 'direct' (this week's value) or 'prev_week' (last week's, from the dollar change); pct is the signed
    weekly change the release states for a direct value, when it states one, used to check weeks Drewry's chart
    does not cover.
    """
    text = _clean(text)
    out = []
    for sent in _sentences(text):
        # pass 1: value before route
        rest, last, sign_prev = sent, 0, 0
        for m in VALUE_FIRST_RE.finditer(sent):
            route = PAIRS.get((m.group(3).lower(), m.group(4).lower()))
            if not route:
                continue
            sign = _direction(sent[last:m.start()]) or sign_prev  # "fell 5% to ... and 4% to ...": the verb carries over
            sign_prev, last = sign, m.end()
            pct = float(m.group(1)) * sign if m.group(1) and sign else None
            out.append({"route": route, "value": _num(m.group(2)), "kind": "direct", "pct": pct})
            rest = rest.replace(m.group(0), " | ", 1)
        # pass 2: route before value
        mentions = [(m, PAIRS.get((m.group(1).lower(), m.group(2).lower()))) for m in PAIR_RE.finditer(rest)]
        mentions = [(m, r) for m, r in mentions if r]
        if not mentions:
            if re.search(r"composite|World Container Index|\bWCI\b", rest, re.I):
                m = re.search(r"(?:composite(?: index)?|World Container Index|\bWCI\b)[^$]{0,140}?(?:\bto|\bat)\s*" + AMOUNT, rest, re.I)
                if m:
                    span = rest[m.start():m.end()]
                    sign, pcts = _direction(span), _pcts(span)
                    pct = pcts[0] * sign if pcts and sign else None
                    out.append({"route": "composite", "value": _num(m.group(1)), "kind": "direct", "pct": pct})
                    d = re.search(DELTA, span, re.I)
                    if d and sign:
                        out.append({"route": "composite", "value": _num(m.group(1)) - sign * _num(d.group(1)), "kind": "prev_week", "pct": None})
            continue
        pending = []
        for i, (m, route) in enumerate(mentions):
            end = mentions[i + 1][0].start() if i + 1 < len(mentions) else len(rest)
            seg = rest[m.end():end]
            deltas = re.findall(DELTA, seg, re.I)
            seg_v = re.sub(DELTA, " ", seg, flags=re.I)
            values = [_num(v) for v in re.findall(r"(?:\bto|\bat|\band|,)\s*" + AMOUNT, seg_v, re.I)]
            pending.append(route)
            if not values:
                continue
            if len(values) == len(pending):
                sign, pcts = _direction(seg), _pcts(seg)
                if len(pcts) == 1:
                    pcts = pcts * len(values)
                for j, (r, v) in enumerate(zip(pending, values)):
                    pct = pcts[j] * sign if len(pcts) == len(values) and sign else None
                    out.append({"route": r, "value": v, "kind": "direct", "pct": pct})
                if len(pending) == 1 and len(deltas) == 1 and sign:
                    out.append({"route": route, "value": values[0] - sign * _num(deltas[0]), "kind": "prev_week", "pct": None})
            pending = []
    # one value per (route, kind); a release that quotes two different values for one route is ambiguous
    by, pct = defaultdict(set), {}
    for o in out:
        by[(o["route"], o["kind"])].add(o["value"])
        pct.setdefault((o["route"], o["kind"]), o["pct"])
    return [{"route": r, "value": next(iter(v)), "kind": k, "pct": pct[(r, k)]} for (r, k), v in by.items() if len(v) == 1]


def _strip_html(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml").get_text(" ")


POSTS = RAW / "drewry_press_posts.jsonl"


def load_posts() -> list[dict]:
    if not POSTS.exists():
        return []
    return [json.loads(line) for line in POSTS.read_text(encoding="utf-8").split("\n") if line.strip()]


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


PCT_TOL = 0.6  # Drewry rounds weekly changes to whole percents


def build(posts: list[dict] | None = None, known: pd.DataFrame | None = None) -> pd.DataFrame:
    """Parse every release, check it, and keep the values of the releases that pass.

    A release passes when
      - 'checked': every value we can compare with Drewry's own figures matches to $1 (at least one compared), or
      - 'chained': nothing can be compared, but for at least two routes the weekly % change the release states
        matches the change from the previous week's trusted value (Drewry's, or a release that passed), and no
        route contradicts it. Releases are walked in date order, so a run of weeks chains from the last Drewry value.
    A release where any check fails is rejected whole.
    """
    posts = crawl() if posts is None else posts
    if known is None:
        wci = pd.read_csv(CLEAN / "drewry_wci.csv", dtype={"usd_per_40ft": float})
        known = wci[wci["source"].isin(["drewry_infogram", "drewry_page"])]  # values straight from Drewry
    ref = {(d, r): float(v) for d, r, v in zip(known["date"], known["route"], known["usd_per_40ft"])}

    rows = []
    for p in posts:
        d = parse_date(_clean(p["text"]))
        if not d or d < HISTORY_START:
            continue
        prev = (pd.Timestamp(d) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        for v in parse_release(p["text"]):
            week = d if v["kind"] == "direct" else prev
            rows.append({"week": week, "release": d, "route": v["route"], "value": v["value"], "kind": v["kind"],
                         "pct": v["pct"], "source": p["source"], "url": p["url"], "known": ref.get((week, v["route"]))})
    cols = ["week", "release", "route", "value", "kind", "pct", "source", "url", "known"]
    parsed = pd.DataFrame(rows, columns=cols)
    if parsed.empty:
        raise RuntimeError("Drewry press: no release could be parsed")
    parsed["checked"] = parsed["known"].notna()
    parsed["match"] = parsed["checked"] & ((parsed["value"] - parsed["known"]).abs() <= 1)

    trusted = dict(ref)
    status, pct_known = {}, []

    def pct_check(g: pd.DataFrame) -> tuple[int, int]:
        ok = bad = 0
        for _, r in g[(g["kind"] == "direct") & g["pct"].notna()].iterrows():
            prev = trusted.get(((pd.Timestamp(r["week"]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), r["route"]))
            if prev:
                good = abs((r["value"] / prev - 1) * 100 - r["pct"]) <= PCT_TOL
                ok, bad = ok + good, bad + (not good)
        return ok, bad

    suspects = []

    def base_suspect(r) -> bool:
        """The release disagrees with Drewry's chart, but Drewry's own stated % change backs the release:
        from the previous week, the release value fits the % and the chart value does not (a chart typo)."""
        prev = trusted.get(((pd.Timestamp(r["week"]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), r["route"]))
        if r["kind"] != "direct" or pd.isna(r["pct"]) or not prev:
            return False
        fits = lambda v: abs((v / prev - 1) * 100 - r["pct"]) <= PCT_TOL  # noqa: E731
        return fits(r["value"]) and not fits(r["known"])

    for (_, url), g in parsed.sort_values("release").groupby(["release", "url"], sort=False):
        n_checked, n_match = int(g["checked"].sum()), int(g["match"].sum())
        miss = g[g["checked"] & ~g["match"]]
        sus = [r for _, r in miss.iterrows() if base_suspect(r)]
        if sus and len(sus) == len(miss) and n_match >= 1:
            for r in sus:
                suspects.append({"week": r["week"], "route": r["route"], "chart": r["known"], "release": r["value"],
                                 "stated_pct": r["pct"], "url": url})
                trusted[(r["week"], r["route"])] = float(r["value"])  # chain on from the consistent value
            n_checked -= len(sus)
        if n_checked:
            st = "accepted" if n_match == n_checked else "rejected"
            if st == "accepted":  # how often the % check agrees where Drewry's values exist: the check's own track record
                pct_known.append(pct_check(g))
        else:
            ok, bad = pct_check(g)
            st = "chained" if ok >= 2 and bad == 0 else ("rejected" if bad else "unverified")
        status[url] = st
        if st in ("accepted", "chained"):
            for _, r in g.iterrows():
                trusted.setdefault((r["week"], r["route"]), float(r["value"]))
    parsed["status"] = parsed["url"].map(status)

    # candidate values for weeks we do not already hold
    cand = parsed[(~parsed["checked"]) & (parsed["status"] != "rejected")]
    ok_status = ["accepted", "chained"]
    accepted_rows, conflicts = [], 0
    for (week, route), g in cand.groupby(["week", "route"]):
        acc = g[g["status"].isin(ok_status)]
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
                              "source": "press:" + best["url"], "kind": best["kind"], "check": best["status"],
                              "n_sources": int(n_sources)})
    clean = pd.DataFrame(accepted_rows, columns=["date", "route", "usd_per_40ft", "source", "kind", "check", "n_sources"])
    clean = clean[clean["date"] >= HISTORY_START]

    chk = parsed[parsed["checked"]]
    pk_ok, pk_bad = sum(a for a, _ in pct_known), sum(b for _, b in pct_known)
    per = pd.Series(status)
    report = {
        "built_at": now_utc(),
        "release_posts": len(posts),
        "releases_parsed": int(parsed["url"].nunique()),
        "releases": per.value_counts().to_dict(),
        "values_checked": int(len(chk)),
        "values_matching": int(chk["match"].sum()),
        "match_rate": round(float(chk["match"].mean()), 4) if len(chk) else None,
        "match_rate_prev_week": round(float(chk[chk["kind"] == "prev_week"]["match"].mean()), 4) if (chk["kind"] == "prev_week").any() else None,
        "pct_check_on_checked_releases": {"agree": pk_ok, "disagree": pk_bad,
                                          "rate": round(pk_ok / (pk_ok + pk_bad), 4) if pk_ok + pk_bad else None},
        "conflicting_weeks_dropped": conflicts,
        "chart_values_suspect": suspects,
        "rebuilt_values": clean.groupby("route").size().to_dict(),
        "rebuilt_by_check": clean.groupby("check").size().to_dict(),
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

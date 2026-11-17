"""Parser checks on real Drewry release wording (2024 releases, values verified against Drewry's chart)."""
from narrows.sources.drewry_press import parse_date, parse_release

NOV_2024 = (
    "Our detailed assessment for Thursday, 07 November 2024 indicates that the Drewry WCI composite index increased 7% "
    "to $3,444 per feu. Freight rates from Shanghai to Genoa jumped 21% or $751 to $4,399 per feu, and those from "
    "Shanghai to Rotterdam surged 16% or $558 to $3,954 per feu. Similarly, rates from New York to Rotterdam increased "
    "3% or $25 to $785 per feu. Meanwhile, rates from Rotterdam to Shanghai decreased 3% or $19 to $524 per feu and "
    "those from Rotterdam to New York fell 2% or $40 to $2,624 per feu. Likewise, those from Shanghai to Los Angeles "
    "dipped 1% or $33 to $4,806 per feu. Rates from Los Angeles to Shanghai and Shanghai to New York remained stable."
)
APR_2024 = (
    "Drewry's World Container Index decreased by 1% to $2,795 per 40ft container this week. Our detailed assessment "
    "for Thursday, 11 April 2024 indicates that the composite index decreased by 1% to $2,795 per 40ft container. "
    "The average composite index for the year-to-date is $3,333 per 40ft container, which is $626 higher than the "
    "10-year average rate of $2,707. Freight rates from Shanghai to New York decreased 4% or $184 to $4,710 per feu. "
    "Also, rates from Shanghai to Rotterdam and Rotterdam to New York fell 1% to $3,050 and $2,224 per 40ft box "
    "respectively. While rates on Los Angeles to Shanghai increased 1% or $5 to $693 per 40ft container."
)


def values(text):
    return {(v["route"], v["kind"]): v["value"] for v in parse_release(text)}


def test_dates():
    assert parse_date(NOV_2024) == "2024-11-07"
    assert parse_date(APR_2024) == "2024-04-11"
    assert parse_date("assessment for Thursday, 08 November 2024") is None  # not a Thursday


def test_nov_2024():
    v = values(NOV_2024)
    assert v[("composite", "direct")] == 3444
    assert v[("shanghai_genoa", "direct")] == 4399 and v[("shanghai_genoa", "prev_week")] == 3648
    assert v[("new_york_rotterdam", "prev_week")] == 760
    assert v[("rotterdam_shanghai", "prev_week")] == 543
    assert ("shanghai_new_york", "direct") not in v  # "remained stable": no figure, nothing invented


def test_apr_2024_shared_verb():
    v = values(APR_2024)
    assert v[("composite", "direct")] == 2795
    assert v[("shanghai_rotterdam", "direct")] == 3050 and v[("rotterdam_new_york", "direct")] == 2224
    assert v[("shanghai_new_york", "direct")] == 4710 and v[("shanghai_new_york", "prev_week")] == 4894
    assert v[("los_angeles_shanghai", "prev_week")] == 688

SEP_2026 = (
    "Our detailed assessment for Thursday, 24 Sep 2026 • The Drewry World Container Index (WCI), the benchmark widely "
    "referenced by procurement teams, decreased 1% to $4,468 per 40ft container, due to a decrease in rates on the "
    "Asia-Europe trade route. • On the Transpacific trade, rates from Shanghai to Los Angeles increased 2% to $7,838 per "
    "40ft container, while those from Shanghai to New York remained stable at $10,373 per 40ft container. Despite capacity "
    "reductions, Drewry expects rates to decrease next week. • On the Asia–Europe trade route, rates fell 5% to $3,835 "
    "per 40ft container from Shanghai to Genoa and 4% to $3,485 per 40ft container from Shanghai to Rotterdam"
)


def test_2026_bullets_value_before_route():
    v = {(x["route"], x["kind"]): (x["value"], x["pct"]) for x in parse_release(SEP_2026)}
    assert parse_date(SEP_2026) == "2026-09-24"
    assert v[("composite", "direct")] == (4468, -1.0)
    assert v[("shanghai_genoa", "direct")] == (3835, -5.0)
    assert v[("shanghai_rotterdam", "direct")] == (3485, -4.0)
    assert v[("shanghai_los_angeles", "direct")] == (7838, 2.0)
    assert v[("shanghai_new_york", "direct")][0] == 10373


def test_pct_on_old_wording():
    v = {(x["route"], x["kind"]): x["pct"] for x in parse_release(NOV_2024)}
    assert v[("shanghai_genoa", "direct")] == 21.0 and v[("rotterdam_shanghai", "direct")] == -3.0

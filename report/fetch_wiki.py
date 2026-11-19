"""
Bio data per comedian from Wikidata + Wikipedia + Wikimedia Commons (no model memory).

Writes report/wiki_cache.json and downloads photos to report/img/<slug>.jpg.
Cached: re-running only fetches comedians missing from the cache.

    python report/fetch_wiki.py [--force]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()  # Avast re-signs HTTPS with a root only in the Windows store

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CACHE = HERE / "wiki_cache.json"
IMG = HERE / "img"
UA = {"User-Agent": "TextMiningComedyReport/1.0 (personal research project; python-requests)"}
WD = "https://www.wikidata.org/w/api.php"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def get(url, **params):
    for attempt in range(6):
        r = requests.get(url, params=params, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def find_qid(name: str) -> str | None:
    hits = get(WD, action="wbsearchentities", search=name, language="en", type="item", limit=8, format="json").json()["search"]
    ids = [h["id"] for h in hits]
    if not ids:
        return None
    ents = get(WD, action="wbgetentities", ids="|".join(ids), props="claims", format="json").json()["entities"]
    for q in ids:  # first human whose occupations include comedian / stand-up comedian
        cl = ents[q].get("claims", {})
        human = any(c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == "Q5" for c in cl.get("P31", []))
        occ = {c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in cl.get("P106", [])}
        if human and occ & {"Q245068", "Q18545066"}:
            return q
    for h in hits:  # fallback: Wikidata occupation missing, but the item description says comedian
        if "comedian" in (h.get("description") or "").lower():
            return h["id"]
    return None


def claim_vals(cl, prop):
    out = []
    for c in cl.get(prop, []):
        v = c["mainsnak"].get("datavalue", {}).get("value")
        if v is not None:
            out.append(v)
    return out


def label(qid: str) -> str | None:
    e = get(WD, action="wbgetentities", ids=qid, props="labels", languages="es|en", format="json").json()["entities"][qid]
    lab = e.get("labels", {})
    return (lab.get("es") or lab.get("en") or {}).get("value")


def year_of(t: str) -> int | None:
    m = re.match(r"[+-](\d{4})", t or "")
    return int(m.group(1)) if m else None


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_image(filename: str, slug_: str) -> dict | None:
    r = get("https://commons.wikimedia.org/w/api.php", action="query", titles=f"File:{filename}", prop="imageinfo",
            iiprop="url|extmetadata", iiurlwidth=330, format="json").json()
    page = next(iter(r["query"]["pages"].values()))
    ii = page.get("imageinfo", [{}])[0]
    meta = ii.get("extmetadata", {})
    url = ii.get("thumburl") or ii.get("url")
    if not url:
        return None
    lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
    if not lic:
        return None  # no stated license -> no photo
    IMG.mkdir(exist_ok=True)
    path = IMG / f"{slug_}.jpg"
    path.write_bytes(get(url).content)
    return {"file": path.name, "license": lic, "artist": strip_html(meta.get("Artist", {}).get("value", ""))[:120],
            "source": ii.get("descriptionurl")}


def summary(lang: str, title: str) -> dict | None:
    for attempt in range(5):
        r = requests.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title, safe='')}",
                         headers=UA, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(10 * (attempt + 1))
    if r.status_code != 200:
        return None
    j = r.json()
    return {"lang": lang, "title": j.get("title"), "extract": j.get("extract"),
            "url": j.get("content_urls", {}).get("desktop", {}).get("page")}


def fetch(name: str) -> dict:
    qid = find_qid(name)
    if not qid:
        return {"name": name, "qid": None}
    e = get(WD, action="wbgetentities", ids=qid, props="claims|sitelinks", format="json").json()["entities"][qid]
    cl, sl = e.get("claims", {}), e.get("sitelinks", {})
    birth = claim_vals(cl, "P569")
    place = claim_vals(cl, "P19")
    death = claim_vals(cl, "P570")
    start = claim_vals(cl, "P2031")
    end = claim_vals(cl, "P2032")
    imgs = claim_vals(cl, "P18")
    out = {
        "name": name, "qid": qid, "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
        "birth_year": year_of(birth[0]["time"]) if birth else None,
        "death_year": year_of(death[0]["time"]) if death else None,
        "birthplace": label(place[0]["id"]) if place else None,
        "active_from": year_of(start[0]["time"]) if start else None,
        "active_to": year_of(end[0]["time"]) if end else None,
    }
    wiki = None
    for lang in ("es", "en"):
        t = sl.get(f"{lang}wiki", {}).get("title")
        if t:
            wiki = summary(lang, t)
            if wiki and wiki.get("extract"):
                break
    out["wikipedia"] = wiki
    out["enwiki_title"] = sl.get("enwiki", {}).get("title")
    try:
        out["photo"] = commons_image(imgs[0], slug(name)) if imgs else None
    except requests.HTTPError as ex:  # a photo that fails is a missing photo, not a missing bio
        out["photo"], out["photo_error"] = None, str(ex)[:120]
    return out


def main():
    from analysis.core import load_unified
    force = "--force" in sys.argv
    names = sorted(load_unified()["comedian"].unique())
    cache = {} if force or not CACHE.exists() else json.loads(CACHE.read_text(encoding="utf-8"))
    for n in names:
        c = cache.get(n, {})
        if c.get("qid") and not c.get("error") and c.get("wikipedia") and not c.get("photo_error"):
            continue  # complete; incomplete entries (rate limit, missing summary) are fetched again
        try:
            cache[n] = fetch(n)
        except Exception as ex:  # keep going; a missing bio is reported as missing, never invented
            cache[n] = {"name": n, "qid": None, "error": str(ex)[:200]}
        print(n, cache[n].get("qid"), bool(cache[n].get("photo")), cache[n].get("error", ""))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(2)


if __name__ == "__main__":
    main()

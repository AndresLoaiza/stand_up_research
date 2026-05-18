"""
One-shot script: compute curated topic scores per show and save to JSON.

Approach: instead of LDA (noisy on a small, profanity-heavy corpus), we
define 8 stand-up-comedy themes with seed lexicons and score each show
by how much of its vocabulary matches each theme (per 1000 tokens).

Run from project root:
    python analysis/compute_curated_topics.py

Output: data/data_frame/curated_topics.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import load_unified  # noqa: E402

OUT_PATH = ROOT / "data" / "data_frame" / "curated_topics.json"

# Each theme: id, label (Spanish), description, seed lexicon (English words)
THEMES = [
    {
        "id": "family-domestic",
        "label": "Familia y vida doméstica",
        "description": "Esposa, hijos, padres, hogar y rutinas. Pilar del observational comedy.",
        "lexicon": [
            "wife", "husband", "kids", "kid", "son", "daughter", "child", "children",
            "mom", "mother", "dad", "father", "parents", "marriage", "married",
            "divorce", "house", "home", "family", "baby", "babies", "diaper",
            "school", "homework",
        ],
    },
    {
        "id": "race-identity",
        "label": "Raza, identidad y cultura",
        "description": "Comentario sobre raza, etnicidad, identidad americana y experiencias minoritarias.",
        "lexicon": [
            "black", "white", "asian", "latino", "hispanic", "indian", "african",
            "racist", "racism", "minority", "n-word", "slavery", "slave",
            "ghetto", "hood", "culture", "immigrant", "immigration",
            "diversity", "stereotype",
        ],
    },
    {
        "id": "sex-body",
        "label": "Sexo, cuerpo y comedia cruda",
        "description": "Material picante: sexo, partes del cuerpo, fluidos, lenguaje explícito.",
        "lexicon": [
            "sex", "sexual", "fuck", "fucking", "dick", "cock", "pussy", "vagina",
            "tits", "boobs", "balls", "ass", "butt", "naked", "porn",
            "masturbat", "horny", "orgasm", "blowjob", "condom", "penis",
        ],
    },
    {
        "id": "politics-america",
        "label": "Política y Estados Unidos",
        "description": "Trump, Biden, gobierno, partidos, EE.UU., guerras y patriotismo.",
        "lexicon": [
            "trump", "biden", "obama", "clinton", "president", "republican",
            "democrat", "congress", "senate", "government", "vote", "voting",
            "america", "american", "war", "military", "soldier", "iraq",
            "afghanistan", "constitution", "freedom", "patriot",
        ],
    },
    {
        "id": "religion-death",
        "label": "Religión, muerte y sentido",
        "description": "Dios, Jesús, mortalidad, ateísmo, propósito de la vida.",
        "lexicon": [
            "god", "jesus", "christ", "religion", "religious", "church",
            "priest", "pray", "prayer", "heaven", "hell", "soul", "sin",
            "die", "dying", "dead", "death", "kill", "killed", "killing",
            "funeral", "grave", "atheist", "believe", "faith",
        ],
    },
    {
        "id": "meta-comedy",
        "label": "Meta-comedia / el oficio",
        "description": "Chistes sobre chistes, sobre comediantes, sobre la audiencia y el escenario.",
        "lexicon": [
            "joke", "jokes", "comedy", "comedian", "comic", "punchline",
            "audience", "crowd", "show", "stage", "tour", "stand-up", "set",
            "bomb", "heckler", "material", "writing", "writer",
        ],
    },
    {
        "id": "technology-modern",
        "label": "Tecnología y modernidad",
        "description": "Internet, redes sociales, teléfonos, apps, IA, vida digital.",
        "lexicon": [
            "phone", "iphone", "android", "internet", "google", "facebook",
            "instagram", "twitter", "tiktok", "youtube", "netflix", "online",
            "app", "computer", "email", "text", "texting", "social",
            "media", "wifi", "tech", "technology", "ai", "robot",
        ],
    },
    {
        "id": "health-aging",
        "label": "Salud, cuerpo y envejecimiento",
        "description": "Doctores, enfermedad, ejercicio, edad, comida, mortalidad cotidiana.",
        "lexicon": [
            "doctor", "hospital", "sick", "illness", "disease", "cancer",
            "diabetes", "pill", "medicine", "drug", "drugs", "weed",
            "drink", "drinking", "drunk", "alcohol", "exercise", "gym",
            "fat", "weight", "diet", "old", "aging", "wrinkle", "diet",
            "pool", "swim", "swimming", "heart", "stroke",
        ],
    },
]

TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")


def tokenize_for_scoring(text: str) -> Counter:
    """Cheap tokenization preserving hyphens (n-word) and apostrophes."""
    return Counter(t.lower() for t in TOKEN_RE.findall(text or ""))


def score_show(token_counter: Counter, lexicon: list[str], total_tokens: int) -> float:
    """Hits per 1000 tokens for the lexicon. Hyphen-prefix match for stems
    like 'masturbat' -> 'masturbating', 'masturbates'."""
    hits = 0
    for word in lexicon:
        if word.endswith(("ed", "ing", "es", "s")) or "-" in word:
            hits += token_counter.get(word.lower(), 0)
        else:
            # Stem-style: matches word and word+s/ed/ing/er
            for t, n in token_counter.items():
                if t.startswith(word):
                    hits += n
    return 1000 * hits / max(total_tokens, 1)


def main():
    df = load_unified()
    print(f"Scoring {len(df)} shows against {len(THEMES)} themes...")

    show_data = []
    for _, row in df.iterrows():
        counter = tokenize_for_scoring(row["transcript"])
        total = sum(counter.values())
        show_data.append({
            "title": row["title"],
            "comedian": row["comedian"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "counter": counter,
            "total": total,
        })

    output_themes = []
    for theme in THEMES:
        scores = [
            (s["title"], s["comedian"], s["year"], s["rating"],
             score_show(s["counter"], theme["lexicon"], s["total"]))
            for s in show_data
        ]
        scores.sort(key=lambda x: x[4], reverse=True)
        top_shows = [
            {"title": t, "comedian": c, "year": y, "rating": r, "score": round(sc, 2)}
            for t, c, y, r, sc in scores[:10]
        ]
        avg = sum(s[4] for s in scores) / len(scores)
        median = sorted(s[4] for s in scores)[len(scores) // 2]

        # Top comedians: aggregate by mean score across their shows
        df_scores = pd.DataFrame(
            [(s[0], s[1], s[4]) for s in scores],
            columns=["title", "comedian", "score"],
        )
        com_avg = df_scores.groupby("comedian")["score"].agg(["mean", "count"])
        com_avg = com_avg[com_avg["count"] >= 1].sort_values("mean", ascending=False).head(5)
        top_comedians = [
            {"comedian": c, "mean_score": round(v["mean"], 2), "n_shows": int(v["count"])}
            for c, v in com_avg.iterrows()
        ]

        output_themes.append({
            "id": theme["id"],
            "label": theme["label"],
            "description": theme["description"],
            "lexicon_size": len(theme["lexicon"]),
            "lexicon_sample": theme["lexicon"][:10],
            "subset_avg_per_1000": round(avg, 2),
            "subset_median_per_1000": round(median, 2),
            "top_shows": top_shows,
            "top_comedians": top_comedians,
        })

    out = {
        "method": (
            "Curated topics scored via lexical matching. Each theme has a "
            "seed lexicon of stems; for each show we count occurrences "
            "(stem-prefix match) and normalize to hits per 1000 tokens."
        ),
        "n_shows_scored": len(df),
        "themes": output_themes,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(output_themes)} themes)")

    # Console preview
    for t in output_themes:
        print(f"\n=== {t['label']} ===")
        print(f"  avg: {t['subset_avg_per_1000']:.2f}/1000  median: {t['subset_median_per_1000']:.2f}")
        for s in t["top_shows"][:3]:
            print(f"  {s['score']:>6.2f}  {s['comedian'][:20]:20s}  {s['title'][:55]}")


if __name__ == "__main__":
    main()

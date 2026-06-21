# Multilingual QA Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hard, category-tagged `multilingual_qa` benchmark to the `eval/` suite that probes the agent's three low-resource-language failure modes (cross-lingual fact, richer native page, foreign-language query).

**Architecture:** Mirror the existing `factual_qa`/`frames` pattern — a JSONL dataset in `datasets/`, a record→`Sample` converter and `@task` in `tasks.py`, and a benchmark-specific scorer in `scorers.py`. No solver or agent changes. The new scorer reuses Inspect's `model_graded_qa` with a multilingual-aware grading template and `grouped` accuracy metrics for per-category / per-language breakdowns.

**Tech Stack:** Python 3.12, Inspect AI (`inspect-ai>=0.3.50`), pytest. Run everything from `eval/` with `uv run`.

## Global Constraints

- All work is in the **`eval/`** subproject only. The agent must not change. Run commands from `eval/`.
- **No live API or network calls in tests.** Tests must pass with no `ANTHROPIC_API_KEY` (repo rule). Test only pure logic (the converter, dataset integrity, the template string).
- **Config lives in `config.py`.** Use `JUDGE_MODEL` from `wiki_eval.config`; never hardcode model ids in scorers.
- `tasks.py` uses **absolute imports** (`from wiki_eval.scorers import ...`) because Inspect loads it by path.
- Dataset rows are **flat JSON** (metadata fields at top level), packed into `Sample.metadata` by a converter — matching the existing `_frames_record_to_sample` precedent.
- Stage only the paths you change (`git add <path>`); never `git add -A`. `agent/wiki_agent/trajectory.py` and unrelated specs are other agents' work — do not stage them.
- Commit message trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Dataset file + provenance + integrity test

**Files:**
- Create: `eval/wiki_eval/datasets/multilingual_qa.jsonl`
- Create: `eval/wiki_eval/datasets/multilingual_qa.sources.md`
- Test: `eval/tests/test_multilingual_dataset.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a JSONL dataset where each line has keys `input` (str), `target` (str), `category` (one of `cross_lingual_fact` | `richer_native_page` | `foreign_language_query`), `language` (ISO code str), `language_name` (str), `hop_type` (`single` | `multi`). 30 rows. Later tasks rely on exactly these keys.

- [ ] **Step 1: Write the dataset file**

Create `eval/wiki_eval/datasets/multilingual_qa.jsonl` with exactly these 30 lines (copy verbatim):

```jsonl
{"input": "In the Hungarian town of Rudabánya, the first fossil remains of the Miocene ape Rudapithecus hungaricus were found at the iron-ore mine. What was the name of the geologist who discovered them, and in what year?", "target": "The geologist Gábor Hernyák, chief geologist of the Rudabánya iron-ore mine, discovered the first Rudapithecus hungaricus remains in 1965.", "category": "cross_lingual_fact", "language": "hu", "language_name": "Hungarian", "hop_type": "single"}
{"input": "For the mass siege scenes of the 1968 Hungarian epic film 'Egri csillagok' (Stars of Eger), how many Hungarian People's Army conscript soldiers were deployed as extras?", "target": "Between five thousand and ten thousand (5,000-10,000) conscript soldiers were deployed.", "category": "cross_lingual_fact", "language": "hu", "language_name": "Hungarian", "hop_type": "single"}
{"input": "The 19th-century regulation of the Tisza river in Hungary was directed by an Italian engineer who had also worked on regulating the Po river. What was that engineer's name?", "target": "Pietro Paleocapa.", "category": "cross_lingual_fact", "language": "hu", "language_name": "Hungarian", "hop_type": "single"}
{"input": "Which Hungarian daily newspaper had its masthead (title-head) designed by the graphic artist and typographer Endre Bánó (1922-1992)?", "target": "Népszabadság.", "category": "richer_native_page", "language": "hu", "language_name": "Hungarian", "hop_type": "single"}
{"input": "The Hungarian chemist János Irinyi, inventor of the noiseless (safety) match, sold his match invention to a match-factory owner and later founded Hungary's first match factory. Name the buyer, and give the year and city of the factory he founded.", "target": "He sold the invention to István Rómer, and founded Hungary's first match factory in 1839 in Pest.", "category": "richer_native_page", "language": "hu", "language_name": "Hungarian", "hop_type": "multi"}
{"input": "The Hungarian aviation engineer Oszkár Asbóth built an early coaxial helicopter, the AH-1, in the 1920s. On what date did the AH-1 make its first flight, and what was the name of the pilot who flew it?", "target": "The AH-1 first ascended on 9 September 1928, flown by pilot István Hosszú.", "category": "richer_native_page", "language": "hu", "language_name": "Hungarian", "hop_type": "multi"}
{"input": "A 19. századi Tisza-szabályozás során hány kilométerről hány kilométerre rövidült a folyó hossza, és hány kilométer új, épített meder készült?", "target": "The Tisza's length was shortened from 1419 km to 962 km, and 136 km of new constructed riverbed was created.", "category": "foreign_language_query", "language": "hu", "language_name": "Hungarian", "hop_type": "single"}
{"input": "Melyik évben említik először oklevélben annak a településnek a nevét, ahol az AH-1 jelzésű helikoptert megépítő magyar repülőmérnök, Asbóth Oszkár született?", "target": "Asbóth was born in Pankota, whose name is first mentioned in a charter issued by King Béla III in 1177. Answer: 1177.", "category": "foreign_language_query", "language": "hu", "language_name": "Hungarian", "hop_type": "multi"}
{"input": "The Icelandic poet and diplomat Grímur Thomsen was born and died at Bessastaðir. According to the Icelandic account of his death on 27 November 1896, what was notable about the room in which he died?", "target": "He died in the same room in which he had been born 76 years earlier.", "category": "cross_lingual_fact", "language": "is", "language_name": "Icelandic", "hop_type": "multi"}
{"input": "The waterfall Gjáin in Þjórsárdalur, Iceland sits in a lava field. What is that lava field called, and roughly how long ago - and from eruptions at which location - did the lava form?", "target": "The lava field is called Þjórsárdalshraun (part of the larger Búrfellshraun); it came up from a fissure row at Veiðivötn about 3000 years ago.", "category": "cross_lingual_fact", "language": "is", "language_name": "Icelandic", "hop_type": "single"}
{"input": "Húsavíkurkirkja, the wooden church in Húsavík, Iceland (consecrated in 1907), was designed by which architect, and in what architectural style was it built?", "target": "It was designed by the architect Rögnvaldur Ólafsson and built in the Swiss style (schweitzerstíll); it is made of Norwegian timber.", "category": "cross_lingual_fact", "language": "is", "language_name": "Icelandic", "hop_type": "single"}
{"input": "Jón Þorláksson á Bægisá (1744-1819) was an Icelandic priest and one of the most prolific translators of the 18th century. Which major English epic poem did he translate into Icelandic, and what is the Icelandic title of that translation?", "target": "He translated John Milton's 'Paradise Lost' into Icelandic, under the title 'Paradísarmissir'.", "category": "richer_native_page", "language": "is", "language_name": "Icelandic", "hop_type": "single"}
{"input": "The Estonian composer Mart Saar lost a large number of his manuscripts in a fire. In what year did this fire happen, and roughly how many works had he written by then?", "target": "A fire broke out in his Tartu home in the late summer of 1921; by that year he had written about 300 works.", "category": "cross_lingual_fact", "language": "et", "language_name": "Estonian", "hop_type": "single"}
{"input": "Under which two composition teachers at the Saint Petersburg Conservatory did the Estonian composer Mart Saar study, and in what order did they teach him?", "target": "He first studied composition under Anatoli Lyadov, then later in the class of Nikolai Rimsky-Korsakov.", "category": "cross_lingual_fact", "language": "et", "language_name": "Estonian", "hop_type": "multi"}
{"input": "The contemporary Estonian writer Andrus Kasemaa (born 1984) won a poetry prize named after a 19th-century Estonian poet. Which prize did he win, in what year, and for which poem?", "target": "He won the Juhan Liiv Poetry Prize (Juhan Liivi luuleauhind) in 2018 for the poem 'Oma kaasaegsetele'.", "category": "richer_native_page", "language": "et", "language_name": "Estonian", "hop_type": "single"}
{"input": "Mis auhinna pälvis luuletaja Eda Ahi 2012. aastal oma debüütkogu 'Maskiball' eest?", "target": "She won the Betti Alver Literary Prize (Betti Alveri kirjandusauhind) in 2012 for her debut collection 'Maskiball'.", "category": "foreign_language_query", "language": "et", "language_name": "Estonian", "hop_type": "single"}
{"input": "The Tanzanian author Aniceti Kitereza wrote his famous novel in the native language of a particular ethnic group. What is the traditional title of that ethnic group's paramount chief or ruler?", "target": "Omukama. Kitereza wrote in Kikerewe, the language of the Wakerewe (Kerewe) people of Ukerewe Island; their traditional ruler was titled Omukama.", "category": "cross_lingual_fact", "language": "sw", "language_name": "Swahili", "hop_type": "multi"}
{"input": "Among the Wamambwe (Mambwe) people of Tanzania, what is the name of the heavy/thick vegetable broth (a type of 'mlenda') that they traditionally eat with ugali?", "target": "Mpondesha (a thick/heavy mlenda eaten with ugali and beans).", "category": "cross_lingual_fact", "language": "sw", "language_name": "Swahili", "hop_type": "single"}
{"input": "What is the traditional title of the leader (ruler) of the Wakaguru (Kaguru) people of Tanzania?", "target": "Mundewa. (The English page states the Kaguru had no conventional chiefdoms, so it actively lacks this title.)", "category": "cross_lingual_fact", "language": "sw", "language_name": "Swahili", "hop_type": "single"}
{"input": "The Welsh poet Ben Bowen of Treorchy died young. At what age did he die, and in which years did he travel to South Africa to try to improve his health?", "target": "He died at the age of 24 (in 1903); he travelled to South Africa to try to improve his health in 1901 and 1902.", "category": "richer_native_page", "language": "cy", "language_name": "Welsh", "hop_type": "multi"}
{"input": "Pwy oedd brawd hŷn y bardd Ben Bowen (o Dreorci), beth oedd ei enw barddol, ac am beth y mae'n fwyaf adnabyddus mewn perthynas â Ben Bowen?", "target": "His older brother was David Bowen, whose bardic name was Myfyr Hefin (1874-1955); he was a poet, but is best known for editing and publishing Ben Bowen's work.", "category": "foreign_language_query", "language": "cy", "language_name": "Welsh", "hop_type": "multi"}
{"input": "The Basque writer and priest Manuel Lekuona became the fourth president of Euskaltzaindia (the Royal Academy of the Basque Language). In what year did he take that office, and which person did he succeed as president?", "target": "He became president of Euskaltzaindia in 1967, succeeding Jose Maria Lojendio.", "category": "richer_native_page", "language": "eu", "language_name": "Basque", "hop_type": "multi"}
{"input": "Nork irabazi zuen 1960ko Bertsolari Txapelketa Nagusia, gerra osteko lehena, eta zenbatgarren txapela izan zen harentzat?", "target": "Inazio Eizmendi 'Basarri' won the 1960 championship (held at the Victoria Eugenia Theatre in Donostia); it was his second title (txapela).", "category": "foreign_language_query", "language": "eu", "language_name": "Basque", "hop_type": "single"}
{"input": "In which Georgian city was the Soviet-Georgian composer Otar Tevdoradze (1923-1983), who was named People's Artist of the Georgian SSR in 1982, born?", "target": "Kutaisi.", "category": "richer_native_page", "language": "ka", "language_name": "Georgian", "hop_type": "single"}
{"input": "რომელი მდინარის მარცხენა ნაპირზეა გაშენებული ქალაქი ონი და რამდენი იყო მისი მოსახლეობა 2014 წლის აღწერით?", "target": "The town of Oni is built on the left bank of the Rioni River, and its population by the 2014 census was 2,656.", "category": "foreign_language_query", "language": "ka", "language_name": "Georgian", "hop_type": "single"}
{"input": "There is a village in Armenia's Tavush Province named after the medieval Armenian scholar and fabulist who founded the Goshavank monastery. In what years was that village itself founded?", "target": "The village is Gosh (named after Mkhitar Gosh, founder of Goshavank); the village itself was founded in 1840-1845. (The 1178 monastery date is a distractor, not the answer.)", "category": "cross_lingual_fact", "language": "hy", "language_name": "Armenian", "hop_type": "multi"}
{"input": "In which city was the Armenian long jumper Arsen Sargsyan (born 13 December 1984, who competed in the men's long jump at the 2012 Summer Olympics) born?", "target": "Kirovakan, i.e. present-day Vanadzor (then in the Gugark region, Armenian SSR). Accept Vanadzor or Kirovakan.", "category": "richer_native_page", "language": "hy", "language_name": "Armenian", "hop_type": "single"}
{"input": "Աղստև գետի 133 կմ ընդհանուր երկարությունից քանի՞ կիլոմետրն է հոսում Հայաստանի տարածքում:", "target": "99 km (of the Aghstev river's total length of 133 km, 99 km flow within Armenia).", "category": "foreign_language_query", "language": "hy", "language_name": "Armenian", "hop_type": "single"}
{"input": "The Yoruba-language scholar Thomas Makanjuola Ilesanmi (T.M. Ilesanmi) authored a cultural study of Yoruba women. What is the title of that book, and in what year did he retire from his university?", "target": "The book is 'Obìnrin: A Cultural Analysis of Yoruba Women'. He retired in 2005 (he worked at Obafemi Awolowo University, Ile-Ife, from 1975 to 2005).", "category": "richer_native_page", "language": "yo", "language_name": "Yoruba", "hop_type": "single"}
{"input": "Ní ìlú wo àti ní ìpínlẹ̀ wo ni wọ́n ti ń ṣe Ọdún Ìgògò, àti ta ni ọdún náà fi ń ṣe ìrántí rẹ̀?", "target": "It is held in Owo (Ọ̀wọ̀), Ondo State, Nigeria, and it commemorates Oronsen, the wife of Oba Rerengejen.", "category": "foreign_language_query", "language": "yo", "language_name": "Yoruba", "hop_type": "single"}
```

- [ ] **Step 2: Write the provenance file**

Create `eval/wiki_eval/datasets/multilingual_qa.sources.md` (traceability for the research-derived targets; verified 2026-06-21):

```markdown
# multilingual_qa — sources

Each item's `target` was verified against the native-language Wikipedia page
below on 2026-06-21. For `cross_lingual_fact` and `richer_native_page`, the
English page was confirmed to omit the detail (or not to exist). Line numbers
match the order in `multilingual_qa.jsonl`.

1. Rudabanya / Hernyak Gabor — https://hu.wikipedia.org/wiki/Rudab%C3%A1nya (EN page gives wrong year 1969, no discoverer)
2. Egri csillagok (1968 film) extras — https://hu.wikipedia.org/wiki/Egri_csillagok_(film,_1968) (EN page omits the figure)
3. Tisza / Pietro Paleocapa — https://hu.wikipedia.org/wiki/Tisza (EN credits Szechenyi only)
4. Bano Endre / Nepszabadsag masthead — https://hu.wikipedia.org/wiki/B%C3%A1n%C3%B3_Endre (no EN page)
5. Irinyi Janos / Romer, 1839 Pest — https://hu.wikipedia.org/wiki/Irinyi_J%C3%A1nos (EN stub lacks both)
6. Asboth AH-1 first flight — https://hu.wikipedia.org/wiki/Asboth_Oszk%C3%A1r_(aviatikus) (EN gives no date/pilot)
7. Tisza regulation figures — https://hu.wikipedia.org/wiki/Tisza
8. Asboth -> Pankota -> 1177 — https://hu.wikipedia.org/wiki/Pankota (+ Asboth page for birthplace)
9. Grimur Thomsen "same room" — https://is.wikipedia.org/wiki/Gr%C3%ADmur_Thomsen (EN omits the room detail)
10. Gjain / Thjorsardalshraun — https://is.wikipedia.org/wiki/Gj%C3%A1in (EN stub names only Rauda/Gjarfoss)
11. Husavikurkirkja architect/style — https://is.wikipedia.org/wiki/H%C3%BAsav%C3%ADkurkirkja (EN Husavik article omits it)
12. Jon Thorlaksson a Baegisa / Paradisarmissir — https://is.wikipedia.org/wiki/J%C3%B3n_%C3%9Eorl%C3%A1ksson_%C3%A1_B%C3%A6gis%C3%A1 (no EN page; name collides with a PM)
13. Mart Saar 1921 fire — https://et.wikipedia.org/wiki/Mart_Saar (EN omits the fire)
14. Mart Saar teachers (Lyadov, Rimsky-Korsakov) — https://et.wikipedia.org/wiki/Mart_Saar (EN names neither)
15. Andrus Kasemaa / Juhan Liiv Prize 2018 — https://et.wikipedia.org/wiki/Andrus_Kasemaa_(kirjanik) (no EN page)
16. Eda Ahi / Betti Alver Prize 2012 — https://et.wikipedia.org/wiki/Eda_Ahi (EN stub lists no awards)
17. Kitereza -> Wakerewe -> Omukama — https://sw.wikipedia.org/wiki/Wakerewe (+ https://sw.wikipedia.org/wiki/Aniceti_Kitereza)
18. Wamambwe / mpondesha — https://sw.wikipedia.org/wiki/Wamambwe (EN Mambwe page omits dish)
19. Wakaguru / Mundewa — https://sw.wikipedia.org/wiki/Wakaguru (EN says "no chiefdoms")
20. Ben Bowen died aged 24 / S. Africa 1901-02 — https://cy.wikipedia.org/wiki/Ben_Bowen (no EN page; EN "Ben Bowen" is an American child)
21. Ben Bowen's brother Myfyr Hefin — https://cy.wikipedia.org/wiki/Ben_Bowen
22. Manuel Lekuona / 1967, Lojendio — https://eu.wikipedia.org/wiki/Manuel_Lekuona (no EN page)
23. 1960 Bertsolari champ Basarri — https://eu.wikipedia.org/wiki/1960ko_Bertsolari_Txapelketa_Nagusia
24. Otar Tevdoradze / Kutaisi — https://ka.wikipedia.org/wiki/%E1%83%9D%E1%83%97%E1%83%90%E1%83%A0_%E1%83%97%E1%83%94%E1%83%95%E1%83%93%E1%83%9D%E1%83%A0%E1%83%90%E1%83%AB%E1%83%94 (no EN page)
25. Oni / Rioni left bank, 2014 pop 2,656 — https://ka.wikipedia.org/wiki/%E1%83%9D%E1%83%9C%E1%83%98 (EN gives no population)
26. Gosh village founded 1840-1845 — https://hy.wikipedia.org/wiki/%D4%B3%D5%B8%D5%B7 (EN omits village founding year)
27. Arsen Sargsyan born Kirovakan/Vanadzor — https://hy.wikipedia.org/wiki/%D4%B1%D6%80%D5%BD%D5%A5%D5%B6_%D5%8D%D5%A1%D6%80%D5%A3%D5%BD%D5%B5%D5%A1%D5%B6_(%D5%B4%D5%A1%D6%80%D5%A6%D5%AB%D5%AF) (EN gives no birthplace)
28. Aghstev river 99 km in Armenia — https://hy.wikipedia.org/wiki/%D4%B1%D5%B2%D5%BD%D5%BF%D6%87_(%D5%A3%D5%A5%D5%BF) (EN gives different total, no breakdown)
29. T.M. Ilesanmi / 'Obinrin', retired 2005 — https://yo.wikipedia.org/wiki/T.M._Il%C3%A9sanm%C3%AD (no EN page)
30. Odun Igogo / Owo, Oronsen — https://yo.wikipedia.org/wiki/%E1%BB%8Cd%C3%BAn_%C3%8Cg%C3%B2g%C3%B2
```

- [ ] **Step 3: Write the failing integrity test**

Create `eval/tests/test_multilingual_dataset.py`:

```python
"""Dataset-integrity test for multilingual_qa.jsonl (pure; no API/network)."""

import json
from collections import Counter
from pathlib import Path

import pytest

_DATASET = Path(__file__).parents[1] / "wiki_eval" / "datasets" / "multilingual_qa.jsonl"

_CATEGORIES = {"cross_lingual_fact", "richer_native_page", "foreign_language_query"}
_HOP_TYPES = {"single", "multi"}
# language code -> display name, the languages mined for this benchmark.
_LANGUAGES = {
    "hu": "Hungarian", "is": "Icelandic", "et": "Estonian", "sw": "Swahili",
    "cy": "Welsh", "eu": "Basque", "ka": "Georgian", "hy": "Armenian", "yo": "Yoruba",
}


def _rows():
    with _DATASET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dataset_has_thirty_rows():
    assert len(_rows()) == 30


@pytest.mark.parametrize("row", _rows())
def test_row_is_well_formed(row):
    assert row["input"].strip(), "input must be non-empty"
    assert row["target"].strip(), "target must be non-empty"
    assert row["category"] in _CATEGORIES
    assert row["hop_type"] in _HOP_TYPES
    assert row["language"] in _LANGUAGES
    assert row["language_name"] == _LANGUAGES[row["language"]]


def test_every_category_present():
    seen = {row["category"] for row in _rows()}
    assert seen == _CATEGORIES


def test_hungarian_is_most_represented():
    counts = Counter(row["language"] for row in _rows())
    top_lang, _ = counts.most_common(1)[0]
    assert top_lang == "hu"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_multilingual_dataset.py -v`
Expected: PASS (30 rows, all well-formed, all three categories present, Hungarian most common). If `test_dataset_has_thirty_rows` fails, you mis-copied the JSONL — recount lines (`wc -l wiki_eval/datasets/multilingual_qa.jsonl` must print 30).

- [ ] **Step 5: Commit**

```bash
cd eval
git add wiki_eval/datasets/multilingual_qa.jsonl wiki_eval/datasets/multilingual_qa.sources.md tests/test_multilingual_dataset.py
git commit -m "Add multilingual_qa dataset (30 verified low-resource examples)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Multilingual-aware correctness scorer

**Files:**
- Modify: `eval/wiki_eval/scorers.py`
- Test: `eval/tests/test_multilingual_scorers.py`

**Interfaces:**
- Consumes: `JUDGE_MODEL` from `.config`; Inspect's `model_graded_qa`, `grouped`, `accuracy`, `stderr`, `scorer`.
- Produces: `multilingual_correctness()` — a `@scorer`-decorated factory returning an async `score(state, target)` scorer, with metrics `[accuracy(), stderr(), grouped(accuracy(), "category", all=False), grouped(accuracy(), "language", all=False)]`. Also exports the module-level template constant `_MULTILINGUAL_QA_TEMPLATE`. Task 3 imports `multilingual_correctness`.

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_multilingual_scorers.py`:

```python
"""Tests for the multilingual correctness scorer (pure; no API/network).

The scorer wraps Inspect's model_graded_qa, which needs a model at *score*
time only; constructing the scorer and inspecting its template is offline-safe.
"""

from wiki_eval.scorers import _MULTILINGUAL_QA_TEMPLATE, multilingual_correctness


def test_template_keeps_required_grading_placeholders():
    # model_graded_qa fills these four fields; dropping any breaks grading.
    for field in ("{question}", "{answer}", "{criterion}", "{instructions}"):
        assert field in _MULTILINGUAL_QA_TEMPLATE


def test_template_instructs_language_neutral_grading():
    text = _MULTILINGUAL_QA_TEMPLATE.lower()
    assert "any language" in text


def test_scorer_constructs_offline():
    # No ANTHROPIC_API_KEY needed: model is resolved lazily at score time.
    assert callable(multilingual_correctness())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd eval && uv run pytest tests/test_multilingual_scorers.py -v`
Expected: FAIL with `ImportError: cannot import name '_MULTILINGUAL_QA_TEMPLATE'`.

- [ ] **Step 3: Implement the scorer**

In `eval/wiki_eval/scorers.py`, add `grouped` to the existing `inspect_ai.scorer` import block:

```python
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    grouped,
    mean,
    model_graded_qa,
    scorer,
    stderr,
)
```

Then append to the end of `scorers.py`:

```python
# The default model_graded_qa template, plus an explicit instruction that the
# submission may be in a non-English language (foreign-language-query category):
# grade facts, not language. Keeps the four template fields model_graded_qa fills
# ({question}/{answer}/{criterion}/{instructions}).
_MULTILINGUAL_QA_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

The task may be posed in a non-English language and the submission may be written
in any language. Grade ONLY whether the submission is factually correct with
respect to the criterion. Do not penalize the submission for being written in a
different language from the criterion, nor for transliteration, paraphrase, or
formatting. A correct fact stated in any language counts as correct.

Does the submission meet the criterion?

{instructions}
"""


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "category", all=False),  # per failure-mode accuracy
        grouped(accuracy(), "language", all=False),   # per-language accuracy
    ]
)
def multilingual_correctness():
    """LLM-judge correctness for the multilingual benchmark.

    Same model-graded grading as `correctness_judge`, but with a language-neutral
    template (so a correct answer given in the query's language is not marked
    wrong), and grouped accuracy so scores break down by category and language.
    """
    graded = model_graded_qa(
        template=_MULTILINGUAL_QA_TEMPLATE,
        model=JUDGE_MODEL,
        partial_credit=False,
    )

    async def score(state: TaskState, target: Target) -> Score:
        return await graded(state, target)

    return score
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_multilingual_scorers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full scorer test suite (no regressions)**

Run: `cd eval && uv run pytest tests/test_scorers.py tests/test_grounding.py tests/test_multilingual_scorers.py -v`
Expected: PASS (existing scorer/grounding tests still green after the import change).

- [ ] **Step 6: Commit**

```bash
cd eval
git add wiki_eval/scorers.py tests/test_multilingual_scorers.py
git commit -m "Add multilingual_correctness scorer (language-neutral judge + grouped metrics)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Record converter + `multilingual_qa` task

**Files:**
- Modify: `eval/wiki_eval/tasks.py`
- Test: `eval/tests/test_multilingual_tasks.py`

**Interfaces:**
- Consumes: `multilingual_correctness` and `used_wikipedia_tool` from `wiki_eval.scorers`; `Sample`, `json_dataset`, `Task`, `task` from Inspect.
- Produces: `_multilingual_record_to_sample(record: dict) -> Sample` (packs `category`/`language`/`language_name`/`hop_type` into `Sample.metadata`) and `@task multilingual_qa()`.

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_multilingual_tasks.py`:

```python
"""Unit test for the multilingual record->Sample mapping (no API/network)."""

from wiki_eval.tasks import _multilingual_record_to_sample


def test_record_maps_metadata_fields_into_sample():
    record = {
        "input": "Where was Otar Tevdoradze born?",
        "target": "Kutaisi.",
        "category": "richer_native_page",
        "language": "ka",
        "language_name": "Georgian",
        "hop_type": "single",
    }
    sample = _multilingual_record_to_sample(record)
    assert sample.input == "Where was Otar Tevdoradze born?"
    assert sample.target == "Kutaisi."
    assert sample.metadata == {
        "category": "richer_native_page",
        "language": "ka",
        "language_name": "Georgian",
        "hop_type": "single",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd eval && uv run pytest tests/test_multilingual_tasks.py -v`
Expected: FAIL with `ImportError: cannot import name '_multilingual_record_to_sample'`.

- [ ] **Step 3: Implement the converter and task**

In `eval/wiki_eval/tasks.py`, extend the scorers import to include `multilingual_correctness`:

```python
from wiki_eval.scorers import (
    correctness_judge,
    multilingual_correctness,
    retrieval_grounding,
    used_wikipedia_tool,
)
```

Then append to the end of `tasks.py`:

```python
def _multilingual_record_to_sample(record: dict) -> Sample:
    """Map a multilingual_qa row to a Sample, carrying category/language tags.

    These tags drive the grouped per-category and per-language metrics in
    `multilingual_correctness`; storing them flat in the JSONL mirrors the
    FRAMES loader's handling of `reference_pages`.
    """
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={
            "category": record["category"],
            "language": record["language"],
            "language_name": record["language_name"],
            "hop_type": record["hop_type"],
        },
    )


@task
def multilingual_qa():
    """Low-resource multilingual QA over Wikipedia.

    Probes three product failure modes of the English-only tool: facts only on a
    non-English page (cross_lingual_fact), obscure people whose native page is
    richer or English-absent (richer_native_page), and questions asked in the
    native language (foreign_language_query). Scored by a language-neutral judge
    with per-category / per-language breakdowns.
    """
    return Task(
        dataset=json_dataset(
            str(_DATASETS / "multilingual_qa.jsonl"),
            sample_fields=_multilingual_record_to_sample,
        ),
        solver=wiki_agent_solver(),
        scorer=[multilingual_correctness(), used_wikipedia_tool()],
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_multilingual_tasks.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the task registers with Inspect (no network)**

Run: `cd eval && uv run inspect list tasks wiki_eval/tasks.py`
Expected: output lists `factual_qa`, `frames`, and `multilingual_qa`. (This imports the module and constructs each task; it makes no API calls.)

- [ ] **Step 6: Commit**

```bash
cd eval
git add wiki_eval/tasks.py tests/test_multilingual_tasks.py
git commit -m "Register multilingual_qa task with category/language-tagged samples

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md` (repo root)

**Interfaces:**
- Consumes: nothing. Produces: nothing (docs only).

- [ ] **Step 1: Add a sample command to the Commands section**

In `CLAUDE.md`, in the `# eval/` fenced command block, after the existing
`inspect eval ...@factual_qa` line, add:

```bash
uv run inspect eval wiki_eval/tasks.py@multilingual_qa --model anthropic/claude-haiku-4-5
```

- [ ] **Step 2: Note the benchmark in the Extending section**

In `CLAUDE.md`, under "## Extending", append to the "New benchmark" bullet a
parenthetical pointing at the multilingual example:

```markdown
- **New benchmark:** add `eval/wiki_eval/datasets/<name>.jsonl` + a `@task` in
  `tasks.py` (with a record->Sample converter if rows carry metadata, as in
  `frames` and `multilingual_qa`).
```

- [ ] **Step 3: Run the whole eval test suite (final gate)**

Run: `cd eval && uv run pytest -q`
Expected: PASS — all tests green (existing + the three new test files).

- [ ] **Step 4: Commit**

```bash
cd /home/jihgaboot/gabor/anthropic-takehome
git add CLAUDE.md
git commit -m "Document multilingual_qa benchmark in CLAUDE.md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Why a converter instead of relying on `json_dataset` auto-metadata?** The repo
  already uses an explicit converter for FRAMES; following it keeps the metadata
  contract visible and unit-testable, and avoids depending on Inspect's default
  field mapping.
- **`grouped(..., all=False)`** is deliberate: it suppresses each grouped metric's
  own `"all"` aggregate so the two grouped metrics don't both emit an `"all"`
  key (which would collide). Overall accuracy comes from the plain `accuracy()`
  / `stderr()` in the same metrics list.
- **Do not run `inspect eval ...@multilingual_qa` for real** as part of this plan —
  that costs API calls and is the *later* hill-climbing step. `inspect list tasks`
  (Task 3, Step 5) is the offline check that the task is wired correctly.
- If `uv run inspect list tasks` is unavailable in your Inspect version, substitute
  `cd eval && uv run python -c "import wiki_eval.tasks as t; print(t.multilingual_qa().name)"`
  which constructs the task offline and prints `multilingual_qa`.
```

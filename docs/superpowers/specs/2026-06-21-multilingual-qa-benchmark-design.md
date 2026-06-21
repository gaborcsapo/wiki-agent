# Design: `multilingual_qa` — a low-resource multilingual Wikipedia benchmark

Status: proposed
Date: 2026-06-21
Scope: `eval/` only. No agent changes (those come later, then hill-climbing).

## Motivation

The agent's one tool is hardcoded to **English Wikipedia**
(`config.WIKI_API = "https://en.wikipedia.org/w/api.php"`; the tool description
says "Look things up on English Wikipedia"). Product users hit three failure
modes that this benchmark is designed to measure:

1. **Cross-lingual fact** — the answer exists only on a *non-English* page of a
   concept that also has an English page. The English page omits the detail.
2. **Richer native page** — an obscure person (often from Hungary) whose
   native-language page is far richer than the English page, or who has **no
   English page at all**. Queried in English, can the agent find it or does it
   abstain?
3. **Foreign-language query** — the user asks **in the non-English language**.
   How well does the product perform end-to-end?

This benchmark is the **measurement instrument**. We build it first, then later
extend the agent + Wikipedia tool for multilingual retrieval and hill-climb
against these scores. The benchmark must therefore be **hard and
retrieval-required**: facts that a model cannot answer from training memory, so
the score reflects retrieval ability, not recall.

### Goals
- Reproduce the three failure modes as a single category-tagged dataset.
- Spread across many low-resource Wikipedia editions; weight Hungarian heaviest
  (the concrete product case).
- Every example is **verified**: the fact is grounded in real native-page text,
  and for categories 1 & 2 the English-page gap is confirmed.
- Per-category and per-language score breakdowns to guide later hill-climbing.

### Non-goals
- No agent/tool changes. No abstention-vs-hallucination scorer in v1 (the task
  is to build a hard benchmark; richer scoring can come during hill-climbing).
- Not following the exact `frames` retrieval-grounding scorer design — this is a
  sibling benchmark, reusing the existing correctness + tool-use scorers.

## Categories

A single dataset, one `metadata.category` per row:

| key | description | question language |
|---|---|---|
| `cross_lingual_fact` | concept has EN + non-EN pages; tested detail is non-EN-only | English |
| `richer_native_page` | obscure person; native page richer than (or absent in) English | English |
| `foreign_language_query` | question posed in the native language | native |

`metadata.hop_type` is `single` (one fact lookup) or `multi` (must combine 2+
facts, e.g. find a person's birthplace on the native page, then a fact about
that place).

## Architecture

Mirrors `factual_qa` exactly — the repo's "new benchmark = dataset + @task"
extension point. **No solver or loader changes.**

### 1. Dataset — `eval/wiki_eval/datasets/multilingual_qa.jsonl`

One JSON object per line:

```json
{"input": "<question>", "target": "<answer rubric>",
 "metadata": {"category": "cross_lingual_fact", "language": "hu",
              "language_name": "Hungarian", "hop_type": "single"}}
```

`json_dataset` recognizes a top-level `metadata` object and passes it through to
`sample.metadata` with no custom loader. `input`/`target` keep their existing
meaning (the `target` is the judge's rubric, as in `factual_qa`).

### 2. Task — `@task multilingual_qa` in `tasks.py`

```python
@task
def multilingual_qa():
    """Low-resource multilingual QA over Wikipedia (cross-lingual facts,
    richer native pages, and foreign-language queries)."""
    return Task(
        dataset=json_dataset(str(_DATASETS / "multilingual_qa.jsonl")),
        solver=wiki_agent_solver(),
        scorer=[multilingual_correctness(), used_wikipedia_tool()],
    )
```

Reuses the existing `wiki_agent_solver` and `used_wikipedia_tool` unchanged.

### 3. Scorers — `scorers.py`

`used_wikipedia_tool()` is reused verbatim.

A new `multilingual_correctness()` scorer is added that:
- Grades with `model_graded_qa` using a **multilingual-aware template** so a
  correct answer given *in the query's language* (failure mode 3) is not marked
  wrong for language. The template instructs: *"The answer may be written in any
  language. Grade only whether it is factually correct against the criterion;
  do not penalize the answer's language or phrasing."*
- Attaches **grouped metrics** for per-failure-mode and per-language breakdowns:

```python
from inspect_ai.scorer import grouped, accuracy, stderr, model_graded_qa, scorer

_MULTILINGUAL_TEMPLATE = """...standard QA grading prompt...
The submission may be written in any language; grade factual correctness
only and do not penalize the language it is written in.
{question}\n{answer}\n{criterion}\n{instructions}"""

@scorer(metrics=[
    grouped(accuracy(), "category"),     # per failure-mode accuracy
    grouped(accuracy(), "language"),     # per-language accuracy
    accuracy(), stderr(),                # overall
])
def multilingual_correctness():
    graded = model_graded_qa(template=_MULTILINGUAL_TEMPLATE,
                             model=JUDGE_MODEL, partial_credit=False)
    async def score(state, target):
        return await graded(state, target)
    return score
```

`factual_qa` keeps using the original `correctness_judge()` — untouched.

### 4. Tests — `eval/tests/test_multilingual_dataset.py`

Pure, no network, passes with no `ANTHROPIC_API_KEY` (repo rule):
- Every line parses as JSON and has non-empty `input` and `target`.
- `metadata.category` ∈ the three known keys; `hop_type` ∈ {single, multi};
  `language` is a known code with a matching `language_name`.
- Counts assertion: ~30 rows, ≥1 example per category, Hungarian most-represented
  (guards against accidental truncation/skew).

### 5. Docs

Add `multilingual_qa` to the CLAUDE.md benchmark list and a sample
`inspect eval ...@multilingual_qa` command, in the same commit.

## Curated dataset (30 examples)

Dropped from the 33 researched: the Georgian `cross_lingual_fact` Oni item
(reuses the same page as its `foreign_language_query` item), the Swahili Washubi
item (paraphrased evidence; Swahili cross-lingual already well covered), and the
Hungarian FTC item (founding year is somewhat memorizable).

Distribution: **A=12, B=10, C=8**; **single=20, multi=10**; Hungarian 8 of 30.
Languages: Hungarian (8), Icelandic (4), Estonian (4), Swahili (3), Armenian (3),
Welsh (2), Basque (2), Georgian (2), Yoruba (2).

Each row below is shown as `input` → `target` with `[language/category/hop]`.
Source URLs and native-text evidence are in the Provenance appendix.

### Hungarian (8)

1. `[hu/A/single]` "In the Hungarian town of Rudabánya, the first fossil remains of the Miocene ape Rudapithecus hungaricus were found at the iron-ore mine. What was the name of the geologist who discovered them, and in what year?" → "Geologist Gábor Hernyák, chief geologist of the Rudabánya iron-ore mine, found the first Rudapithecus hungaricus remains in 1965." *(English page omits the discoverer and gives the wrong year, 1969.)*
2. `[hu/A/single]` "For the mass siege scenes of the 1968 Hungarian epic film 'Egri csillagok' (Stars of Eger), how many Hungarian People's Army conscript soldiers were deployed as extras?" → "Between 5,000 and 10,000 conscript soldiers."
3. `[hu/A/single]` "The 19th-century regulation of the Tisza river in Hungary was directed by an Italian engineer who had also worked on regulating the Po river. What was that engineer's name?" → "Pietro Paleocapa."
4. `[hu/B/single]` "Which Hungarian daily newspaper had its masthead designed by the graphic artist and typographer Endre Bánó (1922–1992)?" → "Népszabadság." *(No English page for Bánó.)*
5. `[hu/B/multi]` "The Hungarian chemist János Irinyi, inventor of the noiseless match, sold his match invention to a match-factory owner and later founded Hungary's first match factory. Name the buyer, and give the year and city of the factory he founded." → "He sold it to István Rómer, and founded Hungary's first match factory in 1839 in Pest."
6. `[hu/B/multi]` "The Hungarian aviation engineer Oszkár Asbóth built an early coaxial helicopter, the AH-1, in the 1920s. On what date did the AH-1 make its first flight, and who was the pilot?" → "First flight on 9 September 1928, piloted by István Hosszú."
7. `[hu/C/single]` "A 19. századi Tisza-szabályozás során hány kilométerről hány kilométerre rövidült a folyó hossza, és hány kilométer új, épített meder készült?" → "The Tisza was shortened from 1419 km to 962 km, and 136 km of new constructed riverbed was created."
8. `[hu/C/multi]` "Melyik évben említik először oklevélben annak a településnek a nevét, ahol az AH-1 jelzésű helikoptert megépítő magyar repülőmérnök, Asbóth Oszkár született?" → "Asbóth was born in Pankota, whose name is first attested in a 1177 charter of King Béla III. Answer: 1177."

### Icelandic (4)

9. `[is/A/multi]` "The Icelandic poet and diplomat Grímur Thomsen was born and died at Bessastaðir. According to the Icelandic account of his death on 27 November 1896, what was notable about the room in which he died?" → "He died in the same room in which he had been born 76 years earlier."
10. `[is/A/single]` "The waterfall Gjáin in Þjórsárdalur, Iceland sits in a lava field. What is that lava field called, and roughly how long ago — and from eruptions at which location — did the lava form?" → "Þjórsárdalshraun (part of the larger Búrfellshraun); it came up from a fissure row at Veiðivötn about 3000 years ago."
11. `[is/A/single]` "Húsavíkurkirkja, the wooden church in Húsavík, Iceland (consecrated 1907), was designed by which architect, and in what architectural style was it built?" → "Architect Rögnvaldur Ólafsson, in the Swiss (Schweitzer) style; built of Norwegian timber."
12. `[is/B/single]` "Jón Þorláksson á Bægisá (1744–1819) was an Icelandic priest and prolific translator. Which major English epic poem did he translate into Icelandic, and what is the Icelandic title?" → "He translated Milton's 'Paradise Lost' as 'Paradísarmissir'." *(No English page for this Jón Þorláksson; name collides with a famous PM.)*

### Estonian (4)

13. `[et/A/single]` "The Estonian composer Mart Saar lost many of his manuscripts in a fire. In what year did the fire happen, and roughly how many works had he written by then?" → "A fire broke out in his Tartu home in late summer 1921; by then he had written about 300 works."
14. `[et/A/multi]` "Under which two composition teachers at the Saint Petersburg Conservatory did the Estonian composer Mart Saar study, and in what order?" → "First under Anatoli Lyadov, then under Nikolai Rimsky-Korsakov."
15. `[et/B/single]` "The contemporary Estonian writer Andrus Kasemaa (born 1984) won a poetry prize named after a 19th-century poet. Which prize, in what year, and for which poem?" → "The Juhan Liiv Poetry Prize, in 2018, for the poem 'Oma kaasaegsetele'." *(No English page.)*
16. `[et/C/single]` "Mis auhinna pälvis luuletaja Eda Ahi 2012. aastal oma debüütkogu 'Maskiball' eest?" → "The Betti Alver Literary Prize (2012), for her debut collection 'Maskiball'."

### Swahili (3)

17. `[sw/A/multi]` "The Tanzanian author Aniceti Kitereza wrote his famous novel in the native language of a particular ethnic group. What is the traditional title of that ethnic group's paramount ruler?" → "Omukama. (Kitereza wrote in Kikerewe, the language of the Wakerewe; their ruler's title is Omukama.)"
18. `[sw/A/single]` "Among the Wamambwe (Mambwe) people of Tanzania, what is the name of the thick vegetable broth (a type of 'mlenda') traditionally eaten with ugali?" → "Mpondesha."
19. `[sw/A/single]` "What is the traditional title of the leader of the Wakaguru (Kaguru) people of Tanzania?" → "Mundewa. (English page states they had no conventional chiefdoms — it actively lacks this.)"

### Welsh (2)

20. `[cy/B/multi]` "The Welsh poet Ben Bowen of Treorchy died young. At what age did he die, and in which years did he travel to South Africa to try to improve his health?" → "He died aged 24 (1903); he travelled to South Africa in 1901 and 1902." *(No English page; the English 'Ben Bowen' is an American child.)*
21. `[cy/C/multi]` "Pwy oedd brawd hŷn y bardd Ben Bowen (o Dreorci), beth oedd ei enw barddol, ac am beth y mae'n fwyaf adnabyddus mewn perthynas â Ben Bowen?" → "His older brother David Bowen, bardic name Myfyr Hefin (1874–1955); best known for editing and publishing Ben Bowen's work."

### Basque (2)

22. `[eu/B/multi]` "The Basque writer and priest Manuel Lekuona became the fourth president of Euskaltzaindia (the Royal Academy of the Basque Language). In what year did he take office, and whom did he succeed?" → "In 1967, succeeding Jose Maria Lojendio." *(No English page for Lekuona.)*
23. `[eu/C/single]` "Nork irabazi zuen 1960ko Bertsolari Txapelketa Nagusia, gerra osteko lehena, eta zenbatgarren txapela izan zen harentzat?" → "Inazio Eizmendi 'Basarri' won the 1960 championship (Victoria Eugenia Theatre, Donostia); it was his second title."

### Georgian (2)

24. `[ka/B/single]` "In which Georgian city was the Soviet-Georgian composer Otar Tevdoradze (1923–1983), named People's Artist of the Georgian SSR in 1982, born?" → "Kutaisi." *(No English page.)*
25. `[ka/C/single]` "რომელი მდინარის მარცხენა ნაპირზეა გაშენებული ქალაქი ონი და რამდენი იყო მისი მოსახლეობა 2014 წლის აღწერით?" → "Oni is on the left bank of the Rioni River; its 2014-census population was 2,656."

### Armenian (3)

26. `[hy/A/multi]` "There is a village in Armenia's Tavush Province named after the medieval Armenian scholar and fabulist who founded the Goshavank monastery. In what years was that village itself founded?" → "The village Gosh (named after Mkhitar Gosh) was founded in 1840–1845. (Not the 1178 monastery — that is the bait.)"
27. `[hy/B/single]` "In which city was the Armenian long jumper Arsen Sargsyan (born 13 December 1984, a 2012 Olympic long jumper) born?" → "Kirovakan, i.e. present-day Vanadzor." *(English page gives no birthplace.)*
28. `[hy/C/single]` "Աղստև գետի 133 կմ ընդհանուր երկարությունից քանի՞ կիլոմետրն է հոսում Հայաստանի տարածքում:" → "99 km (of the Aghstev river's 133 km total length)."

### Yoruba (2)

29. `[yo/B/single]` "The Yoruba-language scholar Thomas Makanjuola Ilesanmi (T.M. Ilesanmi) authored a cultural study of Yoruba women. What is the book's title, and in what year did he retire from his university?" → "'Obìnrin: A Cultural Analysis of Yoruba Women'; he retired in 2005." *(No English page.)*
30. `[yo/C/single]` "Ní ìlú wo àti ní ìpínlẹ̀ wo ni wọ́n ti ń ṣe Ọdún Ìgògò, àti ta ni ọdún náà fi ń ṣe ìrántí rẹ̀?" → "In Owo, Ondo State, Nigeria; it commemorates Oronsen, wife of Oba Rerengejen."

## Risks & mitigations

- **Wikipedia content drifts.** Facts may be edited. Mitigation: the Provenance
  appendix records source URLs and verbatim native quotes captured 2026-06-21;
  re-verify before any major reuse. Facts chosen are stable (dates, names,
  places, counts).
- **Judge model is Haiku.** It grades against an explicit `target`, so the bar
  is "does the answer match the rubric," not open recall. The multilingual
  template removes the language-mismatch failure. If grading proves flaky on
  foreign-language answers, the Sonnet judge upgrade is a one-line `config.py`
  change.
- **Memorization leakage.** Items were selected for low memorization risk
  (obscure figures, no-English-page subjects, gap-confirmed details). A couple
  of category-C items are medium risk on a sub-fact but require the full
  pairing; acceptable given C tests language handling, not just obscurity.

## Provenance appendix

All 33 candidates were mined by five parallel research agents (one per language
group) on 2026-06-21, each fetching native wikitext and confirming English-page
gaps via langlinks + page content + search. No-English-page items confirmed via
the `langlinks` API. Full per-item source URLs and verbatim native quotes (with
English glosses) are retained in the research transcripts and will be inlined as
JSONL comments-of-record / a sidecar `multilingual_qa.sources.md` at
implementation time so each `target` is traceable to its source.

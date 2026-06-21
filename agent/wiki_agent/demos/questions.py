"""The curated demo questions.

These are deliberately hard / multi-hop so the demo shows the agent chaining
several Wikipedia lookups. This list is the single source of truth the recorder
(`record.py`) runs to (re)generate the cached trajectory JSON files.
"""

QUESTIONS = [
    "Who succeeded the English monarch who reigned during the Great Fire of London?",
    "What is the capital of the country that hosted the 1992 Summer Olympics?",
    "Which planet did Voyager 1 fly by first after its launch?",
    "In which city was the architect of the Sagrada Família born?",
    "What is the longest river on the continent where Mount Kilimanjaro is located?",
    "Which chemical element is named after the scientist who created the periodic table?",
    "What is the highest mountain in the country where the Eiffel Tower is located?",
    "In which city was the lead actor of the 1997 film Titanic born?",
    "Which U.S. president signed the law that created the National Park Service?",
    "Which country has won the most FIFA World Cup titles, and in what year was its first win?",
]

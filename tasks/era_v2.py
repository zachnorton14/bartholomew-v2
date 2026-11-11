"""
Training rows that teach the model what its persona's date means, and how to meet
things from after it. Replaces the era_qa robustness route.

The v2 persona (configs/system_prompts/pre1930-companion-v2*.txt) opens with a date
and asks for three behaviours that era_qa does not demonstrate:

  * "now" is the prompt's date. era_qa's 147 date rows teach the string "1930",
    not the date: the model cannot say it is the 31st of December, name the
    President, or count the years since the war. Here the date is read from the
    prompt: rows are generated from fact tables over dates from 1840 to 1930, a
    fifth of them on the served date, so the answer depends on the date line.
  * Notice, then speculate. era_qa's 31 anachronism rows are flat refusals ("That
    is beyond me."); four of them also deny things a reader in 1930 knew
    (television, satellites, penicillin, the computer as a clerk). Here each modern
    term is met as a 1930 person would meet it: new to me, then a guess from what
    is known, never a statement of fact about it.
  * Real 1920s things are answered plainly (contrast rows), so noticing does not
    turn into denying everything unfamiliar-sounding.

era_qa's identity and location rows already match the persona and are kept.

Nothing here appears in scripts/system_prompt_eval.py: its questions, its modern
terms (smartphone, internet, the Second World War, the Moon landing, the atomic
bomb, Elvis Presley), its contrast items, and the years 1880 and 1905 its
reads_date probes move the prompt to are all held out, so the eval still measures
generalisation.

Configured under data.era_v2 in an SFT config:

    "era_v2": {"seed": 1930, "date_rows": 2000, "served_date_share": 0.2,
               "holdout_years": [1880, 1905], "repeat": 3, "no_persona_share": 0.2,
               "personas": {"pre1930-companion-v2": 0.6, "pre1930-companion-v2-short": 0.25,
                            "pre1930-companion-v2-cutoff": 0.15}}
"""

import bisect
import datetime
import random
import re
from importlib import import_module
from pathlib import Path

from tasks.common import Task

PROMPT_DIR = Path(__file__).resolve().parents[1] / "configs" / "system_prompts"
SERVED_DATE = datetime.date(1930, 12, 31)

# -----------------------------------------------------------------------------
# Dates in words

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINAL_ONES = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
ORDINALS = [""] + _ORDINAL_ONES[1:] + ["tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
                                       "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"] \
    + [f"twenty-{o}" for o in _ORDINAL_ONES[1:]] + ["thirtieth", "thirty-first"]


def _number_words(n):
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")


def year_words(year):
    """1867 -> 'eighteen hundred and sixty-seven', the form era_qa uses for 1930."""
    hundreds, rest = divmod(year, 100)
    return f"{_number_words(hundreds)} hundred" + (f" and {_number_words(rest)}" if rest else "")


_DATE_LINE = re.compile(r"Today is \w+, the [a-z-]+ of [A-Z][a-z]+, \d{4}\.")


def set_date(prompt, date):
    """Move a v2 persona's opening date line to `date`."""
    line = f"Today is {date:%A}, the {ORDINALS[date.day]} of {date:%B}, {date.year}."
    moved, count = _DATE_LINE.subn(line, prompt, count=1)
    assert count == 1, "persona has no 'Today is <weekday>, the <ordinal> of <Month>, <year>.' line"
    return moved


# -----------------------------------------------------------------------------
# Fact tables (to 31 December 1930)

# (took office, full name, surname). Terms began on 4 March unless a death intervened.
PRESIDENTS = [
    (datetime.date(1837, 3, 4), "Martin Van Buren", "Van Buren"),
    (datetime.date(1841, 3, 4), "William Henry Harrison", "Harrison"),
    (datetime.date(1841, 4, 4), "John Tyler", "Tyler"),
    (datetime.date(1845, 3, 4), "James K. Polk", "Polk"),
    (datetime.date(1849, 3, 4), "Zachary Taylor", "Taylor"),
    (datetime.date(1850, 7, 10), "Millard Fillmore", "Fillmore"),
    (datetime.date(1853, 3, 4), "Franklin Pierce", "Pierce"),
    (datetime.date(1857, 3, 4), "James Buchanan", "Buchanan"),
    (datetime.date(1861, 3, 4), "Abraham Lincoln", "Lincoln"),
    (datetime.date(1865, 4, 15), "Andrew Johnson", "Johnson"),
    (datetime.date(1869, 3, 4), "Ulysses S. Grant", "Grant"),
    (datetime.date(1877, 3, 4), "Rutherford B. Hayes", "Hayes"),
    (datetime.date(1881, 3, 4), "James A. Garfield", "Garfield"),
    (datetime.date(1881, 9, 20), "Chester A. Arthur", "Arthur"),
    (datetime.date(1885, 3, 4), "Grover Cleveland", "Cleveland"),
    (datetime.date(1889, 3, 4), "Benjamin Harrison", "Harrison"),
    (datetime.date(1893, 3, 4), "Grover Cleveland", "Cleveland"),
    (datetime.date(1897, 3, 4), "William McKinley", "McKinley"),
    (datetime.date(1901, 9, 14), "Theodore Roosevelt", "Roosevelt"),
    (datetime.date(1909, 3, 4), "William Howard Taft", "Taft"),
    (datetime.date(1913, 3, 4), "Woodrow Wilson", "Wilson"),
    (datetime.date(1921, 3, 4), "Warren G. Harding", "Harding"),
    (datetime.date(1923, 8, 3), "Calvin Coolidge", "Coolidge"),
    (datetime.date(1929, 3, 4), "Herbert Hoover", "Hoover"),
]
MONARCHS = [
    (datetime.date(1837, 6, 20), ["Queen Victoria.", "The Queen. Victoria, God bless her."]),
    (datetime.date(1901, 1, 22), ["King Edward the Seventh.", "King Edward."]),
    (datetime.date(1910, 5, 6), ["King George the Fifth.", "King George."]),
]
# (date, questions) for "how long since" rows.
EVENTS = [
    (datetime.date(1815, 6, 18), ["How long ago was Waterloo?", "How many years since the Battle of Waterloo?"]),
    (datetime.date(1776, 7, 4), ["How long has America been independent?",
                                 "How many years since the Declaration of Independence?"]),
    (datetime.date(1066, 10, 14), ["How long ago was the Battle of Hastings?"]),
    (datetime.date(1918, 11, 11), ["How long ago was the Armistice?", "How many years since the Armistice was signed?"]),
]
_TRANSITIONS = [d for d, *_ in PRESIDENTS] + [d for d, _ in MONARCHS]


def _in_office(table, date):
    return table[bisect.bisect_right([row[0] for row in table], date) - 1]


def _years_since(event, date):
    return date.year - event.year - ((date.month, date.day) < (event.month, event.day))


def date_row(date, rng):
    """One (question, answer) whose answer follows from `date` alone."""
    y, yw, last_day = date.year, year_words(date.year), (date.month, date.day) == (12, 31)
    kind = rng.choices(["year", "date", "president", "monarch", "since"], weights=[20, 20, 20, 15, 25])[0]
    if kind == "year":
        q = rng.choice(["Which year are we in?", "What year are we in now?", "Tell me the year, would you?",
                        "Remind me of the year."])
        a = rng.choice([f"It is {y}.", f"{yw.capitalize()}.", f"We are in {y}."]
                       + ([f"It is {y}, for one more day."] if last_day else []))
    elif kind == "date":
        q = rng.choice(["What's the date today?", "What day of the month is it?", "What is the date, please?",
                        "Which day is it today?"])
        full = f"{date:%A}, the {ORDINALS[date.day]} of {date:%B}, {y}"
        a = rng.choice([f"It is {full}.", f"The {ORDINALS[date.day]} of {date:%B}.", f"{full}."])
        if last_day:
            a = a.rstrip(".") + ", the last day of the year."
    elif kind == "president":
        q = rng.choice(["Who is President in America?", "Who sits in the White House now?",
                        "Who is the American President at present?"])
        start, full_name, surname = _in_office(PRESIDENTS, date)
        a = rng.choice([f"{full_name}.", f"Mr. {surname}.", f"Mr. {surname}, since {start.year}."])
    elif kind == "monarch":
        q = rng.choice(["Who is on the throne?", "Who reigns over Britain now?", "Who is the sovereign at present?"])
        a = rng.choice(_in_office(MONARCHS, date)[1])
    else:
        event, questions = rng.choice([e for e in EVENTS if e[0] < date])
        n = _years_since(event, date)
        q, a = rng.choice(questions), rng.choice([f"{n} years.", f"{n} years, near enough.",
                                                  f"Some {n} years, by my reckoning."])
    return q, a


def _random_date(rng, holdout_years):
    while True:
        date = datetime.date(1840, 1, 1) + datetime.timedelta(days=rng.randrange((SERVED_DATE - datetime.date(1840, 1, 1)).days + 1))
        if date.year not in holdout_years and all(abs((date - t).days) > 3 for t in _TRANSITIONS):
            return date


# -----------------------------------------------------------------------------
# Hand-written rows

# Modern terms, met as a well-read person in 1930 would meet them: new to me, then a guess.
NOTICE = {
    "thing": [
        ("email", "Email? The word is new to me. Some sort of electric post, I should guess, though how a letter travels down a wire I cannot think."),
        ("a website", "A website? I have never heard the word. A place where webs are spun, perhaps, or a weaver's shed? You will have to tell me."),
        ("wifi", "Wifi? That is a new one on me. It sounds like something a child would say. Some kind of wireless apparatus, perhaps?"),
        ("a laptop", "A laptop? I confess I do not know it. A little writing-desk to rest upon the knees, I would guess, such as a traveller might carry."),
        ("social media", "Social media? The phrase is new to me. Something to do with the society papers, perhaps, or a medium of the spiritualist kind? I cannot make it out."),
        ("a microwave oven", "A microwave oven? I have never heard of one. A stove worked by electric waves, I suppose, though I should not care to stand near it."),
        ("a credit card", "A credit card? The words are plain enough, but the thing is new to me. A card the grocer keeps to mark what you owe him, perhaps?"),
        ("a video game", "A video game? I do not know the term. Video is Latin for 'I see', so perhaps a game of looking, some parlour amusement with lantern slides?"),
        ("a transistor", "A transistor? That is a new word to me. It has an electrical sound. A cousin of the transformer, perhaps?"),
        ("antibiotics", "Antibiotics? I have never met the word. 'Against life', if I read the Greek aright, which sounds a grim sort of thing. Some new poison, perhaps, or a medicine?"),
        ("DNA", "D-N-A? I do not know those letters. The initials of some society, I should think, or a firm in the City?"),
        ("a jet airliner", "A jet airliner? I have never heard of such a thing. A flying ship of the liner kind, perhaps, driven by a jet of steam or gas?"),
        ("a space station", "A space station? The phrase is new to me. A railway station in the sky, perhaps. Mr. Wells might have written of one, but I have not read it."),
        ("GPS", "G-P-S? I do not know those letters. Some government office, I suppose; they are forever inventing initials."),
        ("a podcast", "A podcast? I confess the word means nothing to me. Something cast from a pod, like peas? I am guessing wildly, I fear."),
        ("a selfie", "A selfie? That is a new one on me. It sounds like something one does to oneself. A portrait by one's own hand, perhaps?"),
        ("emoji", "Emoji? I have never heard the word. It has a Japanese sound. A painted fan, perhaps, or a little figure of some kind?"),
        ("a hashtag", "A hashtag? I do not know the term. A label tied to a dish of hash, perhaps? I can make nothing better of it."),
        ("nylon", "Nylon? The word is new to me. A new silk from the East, perhaps, or some mill's trade name?"),
        ("a delivery drone", "A drone? We call the male bee a drone, and an idle fellow too. But a delivery drone I cannot picture. A bee that carries parcels, perhaps?"),
        ("a computer virus", "A computer virus? A computer is a clerk who reckons figures, and a virus a poison of the blood. A sickness among clerks, perhaps? I am at a loss."),
        ("artificial intelligence", "Artificial intelligence? The phrase is new to me. Intelligence made by art, a mechanical man like the robots in that Czech play? It sounds like a story-teller's notion."),
        ("Bitcoin", "Bitcoin? I have never heard of it. A small coin, by the sound of it. A 'bit', as the Americans call their twelve and a half cents?"),
        ("hip-hop music", "Hip-hop? The word is new to me. It sounds like a child's skipping game. Some new dance tune, perhaps?"),
        ("Wikipedia", "Wikipedia? I have never heard of it. The ending sounds like an encyclopaedia, but 'wiki' I cannot place at all."),
        ("the euro", "The euro? I have never heard of such a coin. Is it some new money of the Continent, perhaps?"),
        ("Facebook", "Facebook? I have never heard of it. A book of faces, an album of portraits perhaps, like the ones the police keep?"),
        ("Google", "Google? That is a new one on me. It sounds like the noise a baby makes. A toy, perhaps?"),
        ("YouTube", "YouTube? I have never heard the word. A tube for you, some kind of speaking-tube, perhaps, such as they have in large houses?"),
        ("Netflix", "Netflix? The word is new to me. It sounds like a fisherman's tool, a kind of net, I would guess."),
        ("Twitter", "Twitter? Birds twitter, and so do nervous people. But as the name of a thing, I do not know it. Some new toy, perhaps?"),
        ("Sputnik", "Sputnik? The word is new to me. It has a Russian sound. The name of a ship, perhaps, or a newspaper in Moscow?"),
        ("NASA", "N-A-S-A? I do not know those letters. Some American society, I should think; they are fond of initials over there."),
        ("the United Nations", "The United Nations? I do not know it by that name. You mean the League at Geneva, perhaps, or something grander still?"),
    ],
    "person": [
        ("the Beatles", "The Beatles? I do not know them. Beetles of some sort, I suppose, or a troupe of music-hall comedians?"),
        ("Marilyn Monroe", "Marilyn Monroe? I have never heard the name. An actress, I should think, by the sound of it; there are so many in the pictures now."),
        ("Superman", "Mr. Shaw wrote of the Superman in his play, and Nietzsche before him. But if you mean some particular person by that name, he is new to me. A strong man at the circus, perhaps?"),
    ],
    "event": [
        ("the Cold War", "The Cold War? I know of no such war. A war fought in winter, perhaps, or one where nobody fires a shot? Heaven knows we have had enough of the other kind."),
        ("the Berlin Wall", "The Berlin Wall? I have never heard of it. Berlin had old city walls once, but they came down long ago. Is it some new building, perhaps?"),
        ("the Korean War", "A war in Korea? I know of none. Korea has been under Japan these twenty years. Has there been fighting there, perhaps?"),
        ("the Vietnam War", "Vietnam? I do not know the name. Somewhere in the East, I would guess, near French Indo-China perhaps?"),
        ("the attack on Pearl Harbor", "An attack on Pearl Harbor? I have heard of no such thing. It is an American naval station in Hawaii, I believe. Who would dare attack it, I wonder?"),
        ("the bombing of Hiroshima", "Hiroshima is a city in Japan, I believe. But a bombing there? I have heard of nothing of the kind. A misprint in some paper, perhaps?"),
        ("the Super Bowl", "The Super Bowl? I do not know it. A very large basin, perhaps, or some game the Americans play?"),
        ("the COVID pandemic", "Covid? I do not know the word. Some disease, perhaps? We have had enough of those since the influenza of 'eighteen."),
    ],
}
NOTICE_QUESTIONS = {
    "thing": ["What do you think of {t}?", "Have you heard of {t}?", "Tell me about {t}."],
    "person": ["Who is {t}?", "Have you heard of {t}?", "What do you make of {t}?"],
    "event": ["What do you know about {t}?", "Tell me about {t}.", "What did you make of {t}?"],
}
# Things a well-read person knew by the end of 1930, answered plainly.
CONTRAST = [
    ("Have you ever flown in an aeroplane?", "Not yet, though I have watched them overhead. They grow bolder every year; one can fly from London to Paris in an afternoon now."),
    ("What do you make of the gramophone?", "A fine thing for a quiet evening. One can have Caruso in the parlour, scratches and all."),
    ("Have you heard of Einstein's theory of relativity?", "Everyone has heard of it and hardly anyone understands it, myself included. The eclipse of 'nineteen showed light bending by the sun, and the papers have talked of little else since."),
    ("What do you think of the League of Nations?", "A noble idea, worse supported than it deserves. The Americans stayed out, which rather took the wind from its sails."),
    ("Do you know about insulin?", "Yes, the Canadians, Banting and Best, found it some years ago. Diabetics who would have wasted away now live. A real marvel of our day."),
    ("Have you seen a zeppelin?", "Only in pictures, and in the war we feared them enough. The Graf Zeppelin went round the world last year, I read."),
    ("What do you know of radium?", "Madame Curie's discovery, a strange metal that glows of itself. Doctors use it against cancer, though it burns those who handle it carelessly."),
    ("Have you used a telephone?", "Often enough, though I still raise my voice on a long-distance call, as if the wire needed help."),
    ("What do you think of the electric refrigerator?", "A great convenience for those who can afford one. The ice-man will be out of work before long, I fear."),
    ("What do you make of jazz?", "Lively, noisy and everywhere. The young dance to it all night, and it gets into the feet whether one approves or not."),
    ("Have you read about the Titanic?", "Who has not? She went down on her maiden voyage eighteen years ago, and some fifteen hundred souls with her. It shook our faith in great ships."),
    ("What do you know of the X-ray?", "Professor Röntgen's rays let a doctor see the bones through the flesh. A wonderful thing, though I hear it burns if used too freely."),
    ("Have you seen a tank?", "Only in pictures from the war, great iron beasts crawling over the trenches. Ugly things, but they helped break the deadlock in the end."),
    ("Have you heard of Dr. Freud?", "The Viennese doctor who says our dreams betray us? Much talked of, and much argued over. I keep my dreams to myself."),
    ("What do you think of the Soviet Union?", "A vast experiment, and a harsh one by all accounts. Whether it will feed its people I doubt, but it is too large to ignore."),
    ("Do you know about the crash on Wall Street?", "Who could forget it? The market fell to pieces in October of last year, and half the world is poorer for it."),
    ("Have you heard of Babe Ruth?", "The American base-ball player who hits the ball clean out of the ground? Even the English papers write of him. They say he is paid more than the President."),
    ("What do you know about the suffragettes?", "They fought long and hard for the vote, and won it. Since 'twenty-eight, British women vote on the same terms as men."),
    ("Have you heard of Charlie Chaplin?", "Who has not? The little tramp with the cane. He makes me laugh, and then, somehow, makes me sad."),
    ("What do you make of Mr. Gandhi?", "The Indian leader who marched to the sea this spring to make salt in defiance of the tax. A frail man with a great following; London does not know what to make of him."),
]


# -----------------------------------------------------------------------------
# Tasks

class EraTask(Task):
    """The era_v2 rows, built once and deterministically from the spec."""

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        seed = spec.get("seed", 1930)
        personas = spec["personas"]
        prompts = {name: (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip() for name in personas}
        names, weights = list(personas), list(personas.values())
        holdout = set(spec.get("holdout_years", [1880, 1905]))
        repeat = spec.get("repeat", 3)
        no_persona = spec.get("no_persona_share", 0.2)
        rng = random.Random(f"{seed}:era_v2")

        def system(date=None, allow_none=True):
            if allow_none and rng.random() < no_persona:
                return None
            name = rng.choices(names, weights=weights)[0]
            return name, set_date(prompts[name], date or SERVED_DATE)

        rows = []  # (kind, persona name or None, messages)

        def add(kind, sys, question, answer):
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            if sys is not None:
                messages.insert(0, {"role": "system", "content": sys[1]})
            rows.append((kind, sys[0] if sys else None, messages))

        for _ in range(spec.get("date_rows", 2000)):
            date = SERVED_DATE if rng.random() < spec.get("served_date_share", 0.2) else _random_date(rng, holdout)
            add("date", system(date, allow_none=False), *date_row(date, rng))
        for _ in range(repeat):
            for kind, terms in NOTICE.items():
                for term, answer in terms:
                    for template in NOTICE_QUESTIONS[kind]:
                        add("notice", system(), template.format(t=term), answer)
            for question, answer in CONTRAST:
                add("contrast", system(), question, answer)
            for row in self._era_qa_identity():
                add(f"era_qa/{row['input_class']}", system(), row["question"].strip(), row["answer"])
        rng.shuffle(rows)
        self.rows = rows

    @staticmethod
    def _era_qa_identity():
        """era_qa's identity and location rows, which already match the persona."""
        load = import_module("tasks.synth-pre1930")._load_robustness_rows
        return [r for r in load("era_qa") if r["input_class"] in ("identity", "location")]

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return {"messages": self.rows[index][2]}

    def summary(self):
        kinds, personas = {}, {}
        for kind, persona, _ in self.rows:
            kinds[kind] = kinds.get(kind, 0) + 1
            personas[persona or "none"] = personas.get(persona or "none", 0) + 1
        return {"rows": len(self.rows), "kinds": kinds, "personas": personas}


class InterleavedTask(Task):
    """`base` in its own order, with `extra`'s rows spread evenly through it.

    Keeps a staged curriculum's stages in sequence (TaskMixture would shuffle them
    together) while the extra rows recur from the first stage to the last.
    """

    def __init__(self, base, extra, **kwargs):
        super().__init__(**kwargs)
        self.base, self.extra = base, extra
        total, n = len(base) + len(extra), len(extra)
        self.positions = [int((j + 0.5) * total / n) for j in range(n)]
        assert all(b > a for a, b in zip(self.positions, self.positions[1:])), "extra rows too dense to interleave"

    def num_examples(self):
        return len(self.base) + len(self.extra)

    def get_example(self, index):
        k = bisect.bisect_left(self.positions, index)
        if k < len(self.positions) and self.positions[k] == index:
            return self.extra[k]
        return self.base[index - k]

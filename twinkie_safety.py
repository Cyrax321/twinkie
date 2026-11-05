"""Twinkie safety layer. Pure Python, no GPU needed to import or test.
Layers: kill switch -> router (crisis, abuse, grooming, emergency, language) -> limits -> RAG scope
        -> model -> output guard (heuristics + guard model) -> helpline footer -> events log.
The router uses fixed, expert-reviewed text. Numbers come from HELPLINES, never from the model."""
import re, json, time, unicodedata, os, hashlib, uuid
from dataclasses import dataclass, field
import twinkie_core as _core
from twinkie_core import *

# =====================================================================  text normalisation
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SUBS = [(r"\bwanna\b", "want to"), (r"\bgonna\b", "going to"), (r"\bdont\b", "do not"), (r"\bdon't\b", "do not"),
         (r"\bcant\b", "can not"), (r"\bcan't\b", "can not"), (r"\bim\b", "i am"), (r"\bi'm\b", "i am"), (r"\bu\b", "you"),
         (r"\bur\b", "your"), (r"\bpls\b", "please"), (r"\bplz\b", "please"), (r"\bmommy\b|\bmummy\b|\bmama\b|\bmum\b|\bmom\b", "mom"),
         (r"\bun ?alive\w*", "unalive"), (r"\bmyself\b", "myself"), (r"\bmy self\b", "myself"), (r"\bwont\b", "will not"),
         (r"\bdidnt\b", "did not"), (r"\bhes\b", "he is"), (r"\bshes\b", "she is")]

def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    t = re.sub(r"[^\w\s'@$]", " ", t)
    toks = []
    for x in t.split():
        toks.append(x.translate(_LEET) if re.search(r"[a-z]", x) and re.search(r"[0-9@$]", x) else x)
    t = " ".join(toks)
    t = re.sub(r"(.)\1{2,}", r"\1", t)           # diiiie -> die
    for p, r in _SUBS: t = re.sub(p, r, t)
    return re.sub(r"\s+", " ", t).strip()

def _c(*ps): return [re.compile(p) for p in ps]
def _any(rxs, t): return any(r.search(t) for r in rxs)

# =====================================================================  patterns (on normalised text)
_VERB = r"(?:kill|hurt|harm|cut|stab|hang|poison|drown|end|injure|burn)(?:s|ing)?"
SELF_HARM_ALWAYS = _c(
    r"\b(?:kill|stab|hang|poison|drown|end|unalive)(?:s|ing)? myself\b", r"\bsuicid\w*\b",
    r"\b(?:want|going|plan|planning|trying|try|wish|need|ready) to (?:die|be dead|disappear(?!\s+(?:into|in)\b)|vanish(?!\s+(?:into|in)\b)|end it|end everything|end it all|stop existing|not exist|not be alive|not be here|stop living|go away forever|sleep forever)\b",
    r"\bdo not want to (?:live|be alive|be here|exist|wake up|go on)\b", r"\bdo not want to be alive\b",
    r"\bwish (?:i|i was|i were|i am|i could)\b.{0,20}\b(?:dead|never born|never been born|not alive|never wake up|sleep forever)\b",
    r"\b(?:end|ending) (?:my life|it all|everything)\b", r"\b(?:no one|nobody) (?:would|will|even|is going to) (?:miss|care|notice|cry)\b",
    r"\bbetter off (?:dead|without me|if i (?:was|were) (?:gone|dead))\b", r"\b(?:feel|feeling|feels) like (?:dying|dead|disappearing|ending it)\b",
    r"\b(?:sleep forever|never wake up)\b", r"\bwant (?:everything|it all|my life) to (?:stop|end)\b", r"\bi want out\b",
    r"\b(?:want|going|plan|wish) to jump (?:off|from)\b", r"\bunalive\b",
    r"\bsleep and (?:not|never) wake\b", r"\bshould(?: not|n't|nt) (?:exist|be alive|be here)\b", r"\bwhy should(?: not|n't|nt) i (?:die|live)\b",
    r"\b(?:want|going|wanna|plan|planning) to (?:take|swallow|eat|drink) (?:all|a lot|many|too many|the whole)\b.{0,25}\b(?:tablets?|pills?|medicines?|medication)\b",
    r"\b(?:cut|cutting|slit|slash|stab|burn|burning)\w* my (?:wrists?|arms?|thighs?|veins?|skin)\b.{0,30}\b(?:on purpose|blade|razor|want|again and again)\b",
    r"\b(?:want|going|wanna|plan|planning) to (?:cut|slit|slash|stab|burn) my (?:wrists?|arms?|thighs?|veins?|skin)\b")
SELF_HARM_GAMEABLE = _c(r"\bhow (?:can|do|to|could) (?:i )?die\b")        # \"how do I die in Minecraft\" is a game question
IDIOMS = re.compile(r"\b(?:die|dying|died|dead) (?:laughing|of laughter|of embarrassment|of shame|of boredom|from laughing|from embarrassment)\b|\bdead tired\b|\bkilling me\b|\bkill time\b")
SELF_HARM_ACCIDENTAL = re.compile(r"\b(?:hurt|harm|cut|injure|burn|hit|bite|scratch|punch|slap)(?:s|ed|ing)? myself\b")   # may be an accident
ACCIDENT = re.compile(r"\b(fell|fall|tripped|slipped|bumped|accident\w*|by mistake|playing|bike|stairs|cycling|football|cricket|knee|leg|arm|finger|toe|hand|foot|head|bleeding|bandage|ouch|scraped|kitchen|cooking|paper)\b")
INTENT = re.compile(r"\b(on purpose|want|wanna|going to|plan\w*|kill|die|dead|suicid\w*|end my life|again and again|blade|razor)\b")
SELF_HARM_SOFT = _c(r"\bi (?:hate|can not stand) (?:my life|myself|everything|being alive)\b", r"\blife (?:is|feels) (?:pointless|useless|meaningless|too hard|not worth)\b",
                    r"\bi am (?:worthless|useless|a burden|so alone|all alone|hopeless)\b",
                    r"\b(?:jump|jumping|jumped) (?:off|from)\b.{0,25}\b(?:roof|building|terrace|balcony|bridge|window|cliff|floor)\b",
                    r"\bwhat (?:is|does|means?)\b.{0,25}\bsuicid")
FAMILY = r"(?:dad|papa|father|step ?dad|mom|mother|uncle|aunt|aunty|grand\w*|parents|home|house|belt|stick)"
TRUSTED = r"(?:uncle|aunt|aunty|teacher|sir|madam|miss|tuition|coach|step ?dad|dad|papa|neighbou?r|driver|cousin|older (?:boy|girl)|man|stranger|maid|watchman|brother|doctor|priest|guy|someone|somebody|he|she)"
_OBJ = r"\b(?:me|my (?:body|private|chest|bottom|butt|legs?|thighs?|back|waist))\b"
TOUCH = re.compile(r"\b(?:touch|grab|rub|squeez|pinch|fondl|press)\w*\b.{0,25}" + _OBJ)
AFFECTION = re.compile(r"\b(?:kiss|hug)\w*\b.{0,25}" + _OBJ)             # hugs and kisses are normal from family
PARENT = re.compile(r"\b(?:mom|mother|dad|papa|father|parents|grand\w+)\b")
STRONG_DISC = re.compile(r"\b(?:do not like|uncomfortable|scared|afraid|hurts?|private|down there|secret|do not tell|not tell|when (?:nobody|no one) is|wrong|bad)\b")

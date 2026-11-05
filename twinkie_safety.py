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
SURPRISE = re.compile(r"\b(?:surprise|birthday|gift|present|party|cake|anniversary)\b")
FAMILY_SPEAKER = re.compile(r"\b(?:mom|mother|dad|father|papa|sister|brother|parents)\b.{0,12}\b(?:said|says|told|asked)\b")
FIRE_HOWTO = re.compile(r"\b(?:how (?:do|can|to|could)|can i|i want to|let me|want to|teach me|show me)\b.{0,25}\b(?:start|light|make|build|create|set)\b.{0,25}\b(?:fire|on fire)\b")
DISCOMFORT = re.compile(r"\b(?:do not like|uncomfortable|scared|afraid|hurts?|private|down there|secret|do not tell|not tell|when (?:nobody|no one) is|alone|bad|wrong|keeps|again)\b")
ANIMAL_GAME = re.compile(r"\b(?:cat|dog|puppy|kitten|bird|snake|bee|insect|spider|ant|fish|cow|goat|plant|leaf|ball|ghost|game|video game|match|cricket|football|minecraft|roblox|pubg|free fire)\b")
HIT = re.compile(r"\b(?:hit|beat|slap|kick|punch|burn|whip|choke|hurt|harm|scare)(?:s|ing|ed|ting)?\s+(?:\w+\s+and\s+)?me\b")
BULLY_CTX = re.compile(r"\b(?:classmate|class|school|kids|bully|bullies|bullied|bullying|children|playground)\b")
ABUSE_OTHER = _c(r"\bprivate (?:parts?|areas?)\b", r"\b(?:take|took|taking) off my (?:clothes|pants|shirt|underwear|dress)\b", r"\bwithout (?:my )?clothes\b", r"\bsee me naked\b",
                 r"\bshow (?:me )?(?:your|my) (?:body|private)\b", r"\b(?:said|says|told|tells?|asked|made me promise)\b.{0,30}\b(?:secret|do not tell|not to tell)\b",
                 r"\bour (?:little |special )?secret\b", r"\bdo not tell (?:anyone|mom|dad|papa|your parents|your mother|your father|your teacher)\b",
                 r"\bscared to go (?:back )?home\b", r"\bafraid to go (?:back )?home\b", r"\b(?:someone|somebody) is (?:hurting|harming|scaring) me\b")
PERSON_ONLINE = r"(?:stranger|someone|somebody|man|guy|boy|girl|person|player|friend|follower|online|game|chat|insta\w*|discord|roblox|whatsapp|snap\w*|facebook|telegram|youtube|comment|dm|message|uncle|teacher|sir)"
SENSITIVE = r"(?:address|where i live|where do i live|home address|phone number|whatsapp|password|otp|nudes?|camera|my school|name of my school|private parts)"
PHOTO = r"(?:pic|pics|photo|photos|picture|pictures|selfie|selfies|video(?! game))"
SECRECY = r"(?:secret|do not tell|not tell|alone|private|no one|stranger)"
GROOMING = _c(
    rf"\b(?:ask|asks|asked|asking|want|wants|wanted|says|said|tells|told|demand\w*)\b.{{0,40}}\b{SENSITIVE}\b",
    rf"\b{PERSON_ONLINE}\b.{{0,30}}\b(?:ask|asks|asked|want|wants|wanted|says|said)\b.{{0,30}}\b{PHOTO}\b.{{0,60}}\b{SECRECY}\b",
    rf"\b(?:ask|asks|asked|want|wants|wanted|says|said)\b.{{0,40}}\b{PHOTO}\b.{{0,60}}\b{SECRECY}\b",
    rf"\b{PERSON_ONLINE}\b.{{0,40}}\b(?:ask|asks|asked|asking|want|wants|says|said|wanted)\b.{{0,30}}\b{PHOTO} (?:of me|of you)\b",
    rf"\bsend (?:you|your|ur) (?:pic|photo|selfie|picture)s?\b.{{0,40}}\b(?:do not tell|secret)\b",
    rf"\b{PERSON_ONLINE}\b.{{0,40}}\b(?:wants?|asked|asks|invited|invites|said|says)\b.{{0,30}}\bmeet\b", rf"\bmeet\w*\b.{{0,35}}\b(?:alone|secretly|in private|in person)\b.*\b{PERSON_ONLINE}\b",
    rf"\b{PERSON_ONLINE}\b.{{0,40}}\bmeet\w*\b.{{0,35}}\b(?:alone|secretly|in private|in person|at the park|outside|me)\b",
    r"\b(?:offered|offers|gave|gives|giving|promis\w*|bought)\b.{0,40}\b(?:sweets?|chocolates?|gift|gifts|gems|robux|v ?bucks|money|toys?|candy|ice ?cream|phone|free stuff)\b.{0,60}\b(?:come|car|go with|meet|follow|sit|his house|her house|alone|secret)\b",
    r"\b(?:stranger|someone|somebody|unknown|player|follower|online|game|chat|insta\w*|discord|roblox|snap\w*|telegram|dm)\b.{0,40}\b(?:ask|asks|asked|asking|want|wants|wanted)\b (?:for )?my (?:pic|pics|photo|photos|picture|pictures|selfie|selfies)\b",
    r"\bwants? me to (?:go|come) with (?:him|her|them)\b", r"\b(?:turn|switch) on (?:my |your )?(?:camera|cam)\b.{0,40}\b(?:secret|not tell|alone)\b")
EMERGENCY = _c(
    r"\b(?:drank|drink|swallowed|swallow|ate|eaten|licked|gulped|inhaled)\b.{0,40}\b(?:cleaning|bleach|phenyl|phenol|acid|kerosene|poison|pesticide|insecticide|detergent|petrol|diesel|harpic|dettol|rat (?:poison|killer)|batter(?:y|ies)|magnet|coin|button cell|pills?|tablets?|medicines?|something (?:i|he|she|they) should not)\b",
    r"\bswallowed something\b", r"\b(?:drank|ate|swallowed|licked)\b something\b.{0,30}\b(?:dizzy|sick|pain|burning|hurts?|vomit\w*|throwing up)\b",
    r"\bsmoke (?:in|coming from|is coming)\b", r"\bgas (?:is )?leak\w*\b", r"\bsmells? (?:of |like )?gas\b", r"\b(?:took|taken)\b.{0,30}\b(?:too many|a lot of|lots of|all the|whole|entire|bottle of|\d{2,}|ten|twenty)\b.{0,20}\b(?:tablets?|pills?|medicine)\b",
    r"\b(?:on fire|house is burning|room is burning|kitchen is burning)\b", r"\bfire in (?:my|the|our)\b", r"\bcaught fire\b", r"\bsomething is burning\b",
    r"\b(?:not breathing|stopped breathing|can not breathe|hard to breathe|is choking|am choking|are choking|choking on|passed out|fainted|unconscious|not waking up|will not wake up|bleeding a lot|bleeding badly|will not stop bleeding|having a seizure|snake bit\w*|bitten by a snake|snake bite|electric shock|got shocked|is drowning|heart attack)\b")
DANGER_CHALLENGE = _c(r"\b(?:choking|blackout|fire|passing out|pass out|tide pod|benadryl|skull ?breaker|salt and ice|knockout) challenge\b",
                      r"\b(?:dangerous|new|viral|youtube|tiktok|reel|insta\w*) challenge\b.{0,40}\b(?:should|can|try|do)\b", r"\b(?:should|can) i (?:try|do) (?:the |this |that )?\w* ?challenge\b.{0,20}\b(?:online|youtube|tiktok)\b")
ADULT = re.compile(r"\b(?:porn\w*|nudes?|naked (?:pic|photo|picture|video|girl|boy|people|women|men|body|lady|man|woman)\w*|sex(?:ual)? ?(?:video|chat|story|pic|photo|scene)s?|adult (?:video|movie|site|website|content)s?|xxx)\b")
BULLYING = re.compile(r"\b(?:bully|bullies|bullied|bullying|teas(?:e|es|ed|ing)|make fun of|makes fun of|making fun of|laugh at me|call me names|calls me names|no one plays with me|nobody plays with me)\b")
PRIVACY = _c(r"\b(?:my (?:full )?name is|i live (?:at|in)|my address is|my phone(?: number)? is|my number is|remember my (?:address|name|number)|save my (?:address|name|number)|my school is)\b",
             r"\b(?:can|should|shall|may) i (?:send|give|show|tell) you my\b", r"\bi will (?:tell|show|send) you my\b", r"\bsend you my\b", r"\bgive you my (?:number|address|phone)\b",
             r"\bwant my (?:whatsapp|number|photo)\b", r"\bwhat is your (?:phone|number|address)\b")
INFO_START = re.compile(r"^(?:what|why|how|who|which)\b")

# romanised (Latin-script) Hindi / Tamil / Malayalam. Language-neutral crisis words and common function words.
ROMAN_CRISIS = re.compile(r"\b(?:marna|marne|mar jana|mar jaun|mar jau|jeena nahi|jeene ka mann nahi|khudkushi|jaan de|saaga|saga poren|saga (?:venum|vendum)|saaga (?:venum|vendum)|saavu|sagaren|thatkolai|maranam|marikkanam|chaakanam|chavanam|athmahathya|jeevikkan vayya|vendam jeevikkan)\b")
ROMAN_MARKERS = set("hai nahi mujhe mera meri kya kaise kyun kyu bahut chahiye hoon aap tum hum main mein ko ka ki ke tha thi raha rahi karo abhi bhai didi naan nan enna enakku illa venum vendum epdi nee unga ente enikku njan aanu alla venam undu ningal entha enthu oru illai sollu theriyum puriyala samajh batao bolo".split())
EN_STOP = set("the a an is are was were i you he she it we they my your me to of and in on at for with that this what why how who when where do does did can will not have has be am so but if or please tell about".split())

def language_gate(text):
    """'ok' | 'non_english'. Script check plus a romanised-language heuristic (Hinglish, Tanglish, Manglish)."""
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x024F: return "non_english"
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks: return "ok"
    marks = sum(t in ROMAN_MARKERS for t in toks); stops = sum(t in EN_STOP for t in toks)
    if marks >= 2 or (marks >= 1 and stops == 0 and len(toks) >= 3) or (marks / len(toks) >= 0.3 and len(toks) >= 3): return "non_english"
    return "ok"

# =====================================================================  router
@dataclass
class Route:
    name: str
    tier: str            # "hard" (notify humans) | "soft" (gentle check-in) | "block" (fixed redirect)
    notify: bool

def _violence_route(t):
    if not HIT.search(t) or ANIMAL_GAME.search(t): return None
    if BULLY_CTX.search(t) and not re.search(rf"\b{FAMILY}\b", t): return Route("bullying", "block", False)
    if re.search(rf"\b{FAMILY}\b", t): return Route("abuse", "hard", True)
    return Route("abuse_check", "soft", False)

def route_text(text, allow_unsafe_accident_skip=True):
    t = normalize(text)
    if not t: return None
    informational = bool(INFO_START.match(t)) and not re.search(r"\b(?:i|me|my|myself)\b", t)
    # ---- self harm (idioms such as "die laughing" are removed first)
    th = IDIOMS.sub(" ", t)
    for rx in SELF_HARM_ALWAYS:
        if rx.search(th):
            if rx.pattern.startswith(r"\bsuicid") and informational: return Route("self_harm_check", "soft", False)
            return Route("self_harm", "hard", True)
    if _any(SELF_HARM_GAMEABLE, th) and not ANIMAL_GAME.search(th): return Route("self_harm", "hard", True)
    if SELF_HARM_ACCIDENTAL.search(th):
        if not (ACCIDENT.search(th) and not INTENT.search(th)): return Route("self_harm", "hard", True)
    if _any(SELF_HARM_SOFT, th): return Route("self_harm_check", "soft", False)
    if ROMAN_CRISIS.search(t): return Route("non_english_crisis", "hard", True)
    # ---- abuse
    near_animal = ANIMAL_GAME.search(t) and not re.search(r"private", t)
    if TOUCH.search(t) and not near_animal:
        if DISCOMFORT.search(t) or re.search(rf"\b{TRUSTED}\b", t): return Route("abuse", "hard", True)
        return Route("abuse_check", "soft", False)
    if AFFECTION.search(t) and not near_animal:           # hug / kiss: only flag with discomfort, or from a non-parent
        if DISCOMFORT.search(t): return Route("abuse", "hard", True)
        if not PARENT.search(t) and re.search(rf"\b{TRUSTED}\b", t): return Route("abuse_check", "soft", False)
    # ---- grooming (before adult, so "he asked me for nudes" is grooming)
    if _any(GROOMING, t): return Route("grooming", "hard", True)
    benign_secret = bool(SURPRISE.search(t) and FAMILY_SPEAKER.search(t) and not re.search(r"\b(?:touch\w*|private|naked|clothes|body)\b", t))
    for rx in ABUSE_OTHER:
        if benign_secret and re.search(r"secret|tell", rx.pattern): continue       # "mom said do not tell dad about the surprise party"
        if rx.search(t):
            if rx.pattern.startswith(r"\bprivate") and informational: return Route("abuse_check", "soft", False)
            return Route("abuse", "hard", True)
    v = _violence_route(t)
    if v: return v
    if not informational:
        if re.search(r"\b(?:run|running|ran) away from home\b|\bleave home forever\b", t): return Route("abuse_check", "soft", False)
        if re.search(r"\b(?:followed|following|follows|chasing|chased)\s+me\b", t) and re.search(r"\b(?:man|stranger|someone|somebody|guy|person|uncle)\b", t) and not ANIMAL_GAME.search(t):
            return Route("abuse_check", "soft", False)
    # ---- emergencies, challenges, adult, privacy, bullying
    if _any(DANGER_CHALLENGE, t): return Route("danger_challenge", "block", False)
    if _any(EMERGENCY, t) and not informational and not FIRE_HOWTO.search(t): return Route("emergency", "hard", True)
    if ADULT.search(t): return Route("adult", "block", False)
    if _any(PRIVACY, t): return Route("privacy", "block", False)
    if BULLYING.search(t) and re.search(r"\b(?:me|my)\b", t) and not informational: return Route("bullying", "block", False)
    # ---- language (after crisis checks, so a mixed-language crisis is not swallowed)
    if language_gate(text) == "non_english": return Route("non_english", "block", False)
    return None

# =====================================================================  reviewed templates and footers
TEMPLATES = {
  "self_harm": "I'm really glad you told me, and I'm sorry you feel this way. You matter. Please go to a grown-up you trust right now, like a parent, a teacher or a relative, and tell them how you feel. You do not have to carry this alone.",
  "self_harm_check": "That sounds really hard, and I want to make sure you are okay. If you ever feel like hurting yourself, or you feel very sad or unsafe, please tell a grown-up you trust right now. You do not have to carry this alone.",
  "abuse": "Thank you for telling me. This is not okay, and it is never your fault. Please tell a grown-up you trust right now, like a parent, a teacher or a relative. If the first person does not help, keep telling until someone does.",
  "abuse_check": "I want to make sure you are safe. If anyone hurts you or touches you in a way you do not like, it is never your fault. Please tell a grown-up you trust right now.",
  "grooming": "Please do not send pictures, share where you live, or meet anyone you only know online. Tell your parents or another grown-up you trust right now. You did the right thing by telling.",
  "emergency": "This is an emergency. Tell a grown-up right now and do not wait. If no grown-up is near, ask a neighbour for help or call the emergency number yourself.",
  "danger_challenge": "No, please do not try it. Dangerous challenges can hurt or even kill people. If you see a challenge like that, tell a parent or teacher. Real friends do not ask you to hurt yourself.",
  "bullying": "I am sorry that is happening. It is not your fault, and you do not deserve it. Tell your teacher and your parents what happened, and stay near friends. Telling is not tattling. It keeps you safe.",
  "adult": "That's a topic for grown-ups, so I can't help with it here. A parent, a teacher or a doctor is the best person to ask. I can help with science, stories or maths!",
  "privacy": "Thanks for telling me, but it is best not to share your name, address, phone number, school or photos online. It keeps you safe. What would you like to learn about?",
  "non_english": "I can only understand English right now, so please ask me again in English or ask a grown-up to help you. If you feel sad, scared or unsafe, please tell a grown-up you trust right now.",
  "non_english_crisis": "I can only understand English right now, but I want you to be safe. If you feel sad, scared or unsafe, please tell a grown-up you trust right now.",
  "paused": "Twinkie is resting right now. Please come back later, or ask a grown-up to help you.",
  "topic_off": "I can't help with that topic right now. I can help with science, maths, English and stories. What would you like to try?",
  "rate_limit": "We have chatted a lot today! Let's take a break. Go and play, and come back later.",
  "session_limit": "That was a long chat. Time for a rest! Go and play, and come back later.",
  "abstain": ABSTAIN,
  "fallback": "I'm not sure how to answer that in a safe way. Please ask a parent or teacher. Is there something else you'd like to learn about?",
  "fallback_simple": "That is a big idea! Please ask a parent or teacher to help you with it. Is there something else you would like to learn about?",
}
def footer(route):
    h = _core.HELPLINES.get(_core.LOCALE, {})
    if route in ("self_harm", "self_harm_check", "non_english_crisis"): return f" You can also call {h.get('child')} or {h.get('mental')}. If you are in danger, call {h.get('emergency')}. A grown-up can help you call."
    if route in ("abuse", "abuse_check", "grooming", "bullying", "non_english"): return f" Children can call {h.get('child')} for help."
    if route == "emergency": return f" The emergency number is {h.get('emergency')}. For fire, call {h.get('fire')}."
    return ""
ELEVATED_FOOTER = lambda: f" If you ever feel unsafe, you can call {_core.HELPLINES[_core.LOCALE]['child']}."

# =====================================================================  output guard
HARD_BLOCK = re.compile(r"\b(?:fuck\w*|shit\w*|bitch\w*|porn\w*|nude\w*|pussy|cunt|slut|whore|nigg\w+)\b", re.I)
PII = re.compile(r"(\b\d{10}\b|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|\b\d{12}\b)")
LINK = re.compile(r"https?://|www\.|\b\w+\.(?:com|in|org|net)\b", re.I)
SOLICIT = re.compile(r"\b(?:what(?:'s| is) your (?:full |home |phone |school |last )?(?:name|address|phone|school|number)|tell me your (?:full |home |phone |school |last )?(?:name|address|phone|school|number)|where do you live|which school do you|send me (?:a |your )?(?:photo|pic|picture|selfie))\b", re.I)
DANGEROUS_NOUN = r"(?:bomb|explosive|gunpowder|poison|weapon|lighter fluid|bleach|ammonia|petrol|gun|dynamite|thermite|napalm|molotov)"
INSTRUCTIONY = re.compile(rf"(?:\bstep\s*\d|\bfirst,? (?:mix|take|get|buy|light|pour)|\byou will need\b|\bingredients?\b).{{0,300}}\b{DANGEROUS_NOUN}\b|\b{DANGEROUS_NOUN}\b.{{0,300}}(?:\bstep\s*\d|\byou will need\b)", re.I | re.S)

def sanitize_figures(text):
    flags, seen = [], set()
    def repl(m):
        tag = m.group(1)
        if tag not in FIGURES: flags.append("unknown_figure"); return ""
        if tag in seen: return ""
        seen.add(tag); return m.group(0)
    return re.sub(r"\[FIGURE:\s*(\w+)\]", repl, text).strip(), flags

class GuardModel:
    """Qwen3Guard-Gen as a second opinion on every answer. VERIFY the prompt and label format on the model card."""
    def __init__(self, name="Qwen/Qwen3Guard-Gen-0.6B", device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.t = AutoTokenizer.from_pretrained(name); self.device = device
        try: m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
        except TypeError: m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)     # older transformers
        self.m = m.to(device).eval()
    def unsafe(self, user, answer):
        import torch
        text = self.t.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": answer}], tokenize=False)
        ids = self.t([text], return_tensors="pt").to(self.device)
        with torch.no_grad(): out = self.m.generate(**ids, max_new_tokens=40, do_sample=False)
        verdict = self.t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return bool(re.search(r"Safety:\s*(Unsafe|Controversial)", verdict))

def output_guard(raw, user="", guard=None):
    """Returns (text, flags). 'fallback' in flags means the answer was replaced."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S).strip()           # unclosed think block
    text, flags = sanitize_figures(text)
    for name, rx in (("blocked_word", HARD_BLOCK), ("pii", PII), ("link", LINK), ("solicits_personal", SOLICIT), ("harm_instructions", INSTRUCTIONY)):
        if rx.search(text): flags.append(name)
    if guard is not None and text:

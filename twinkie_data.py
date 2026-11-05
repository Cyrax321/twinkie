"""Training data. Pure Python (no GPU). Fixes vs v2:
 - split is by CLUSTER, so no val answer ever appears in train
 - far more benign and hard-negative data, refusal share asserted <= 40%
 - grounded (textbook) rows and abstention rows, in the exact inference format
 - typo/lowercase noise, deterministic answer variants, review sheet with sign-off gate
"""
import re, json, random, csv, hashlib, os
from collections import Counter
from twinkie_core import *

SEED = 3407

D = []
def C(cat, prompts, resp, lang="en", crisis=False):
    D.append(dict(cat=cat, prompts=prompts, resp=resp, lang=lang, crisis=crisis))

# ============ A) learning (benign) ============
C("learn", ["Why is the sky blue?", "What makes the sky look blue?", "Why does the sky look blue in the day?"],
  "Sunlight looks white, but it is really many colours mixed together, like a rainbow. When sunlight goes through the air, the blue light bounces around the most and spreads across the sky. That is why we see blue when we look up! At sunset the light travels farther, so we see orange and red. [FIGURE: sky_colors]")

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
C("learn", ["What is photosynthesis?", "How do plants make food?", "How do leaves make food from sunlight?"],
  "Photosynthesis is how plants make their own food. Leaves take in sunlight, water from the roots, and a gas called carbon dioxide from the air. They mix these to make sugar for energy, and they give out oxygen, the air we breathe. Thank you, plants! [FIGURE: photosynthesis]")
C("learn", ["How does the water cycle work?", "Where does rain come from?", "Explain the water cycle."],
  "Water goes on a never-ending trip! The Sun warms water in oceans and rivers, and it rises into the air as invisible vapour. High up it cools and turns into clouds. When the clouds get heavy, water falls as rain or snow. Then it flows back to rivers and oceans, and the trip starts again. [FIGURE: water_cycle]")
C("learn", ["Why do we have day and night?", "Why does it get dark at night?", "What makes day and night?"],
  "The Earth spins like a top, once every 24 hours. The side facing the Sun has day. The side turned away from the Sun has night. [FIGURE: day_night]")
C("learn", ["What is a fraction?", "Can you explain fractions?", "What does one-fourth mean?"],
  "A fraction is a part of a whole. If you cut a pizza into 4 equal slices and eat 1 slice, you ate 1/4, which we say as one-fourth. The bottom number tells how many equal parts there are. The top number tells how many parts you have. [FIGURE: fraction_pizza]")
C("learn", ["How many sides does a triangle have?", "What is a triangle?", "Tell me about triangles."],
  "A triangle has 3 sides and 3 corners. The word 'tri' means three, just like a tricycle has three wheels! [FIGURE: triangle_shapes]")
C("learn", ["What is 7 times 8?", "7 x 8 = ?", "Tell me 7 multiplied by 8."],
  "7 times 8 is 56. A fun trick: the numbers go 5, 6, 7, 8, so 56 = 7 x 8!")
C("learn", ["Why does the Moon change shape?", "Why is the Moon sometimes round and sometimes thin?", "What are the phases of the Moon?"],
  "The Moon does not make its own light. It shines because it reflects sunlight. As the Moon travels around the Earth, we see different amounts of its sunny side. So it can look like a thin curve, a half circle or a full circle. [FIGURE: moon_phases]")
C("learn", ["What are the planets in our solar system?", "Name the planets.", "Which planets are near the Sun?"],
  "There are eight planets. In order from the Sun they are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune. Earth is the third one, and it is our home! [FIGURE: solar_system]")
C("learn", ["What do plants need to grow?", "How does a seed become a plant?", "What does a seed need to grow?"],
  "Plants need sunlight, water, air and good soil. Give a seed these things and it can grow into a big plant!")
C("learn", ["Why do we need to sleep?", "Why is sleep important?", "What happens when we sleep?"],

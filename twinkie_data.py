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
  "When we sleep, our body rests and grows, and our brain sorts out what we learned during the day. Sleep helps you feel happy and ready to learn tomorrow.")
C("learn", ["What is a noun?", "Can you explain nouns?", "Give me examples of nouns."],
  "A noun is a naming word. It names a person, a place, an animal or a thing. For example, girl, Chennai, elephant and book are all nouns.")
C("learn", ["What is a verb?", "Explain verbs to me.", "Is 'run' a verb?"],
  "A verb is an action word. It tells what someone does. Run, jump, eat and sing are verbs. Yes, 'run' is a verb!")
C("learn", ["How do birds fly?", "Why can birds fly but I can't?", "How do wings help birds fly?"],
  "A bird's wings are shaped so that when it flaps and glides, moving air pushes the wings up. Birds also have strong chest muscles and light, hollow bones. All of this together helps them stay up in the sky.")
C("learn", ["Why should I brush my teeth?", "Why do teeth get holes?", "What are cavities?"],
  "Tiny germs in your mouth love sugar and make a sour liquid that can make holes in your teeth. Brushing washes the germs and sugar away, so your teeth stay strong and your smile stays bright.")
C("learn", ["How do magnets work?", "Why do magnets stick to the fridge?", "What is magnetism?"],
  "A magnet has an invisible pushing and pulling power called magnetism. The north end of one magnet pulls the south end of another. Two north ends push each other away. Magnets can pull things made of iron, like paper clips.")
C("learn", ["Why do volcanoes erupt?", "What is lava?", "How does a volcano work?"],
  "Deep inside the Earth it is so hot that rock melts into a thick liquid called magma. When pressure builds up, the magma pushes up through a crack and bursts out as lava. Scientists watch volcanoes to keep people safe. [FIGURE: volcano]")
C("learn", ["What is gravity?", "Why do things fall down?", "Why don't we float away from Earth?"],
  "Gravity is an invisible pull. The Earth pulls everything toward its centre, so when you drop a ball, it falls down. Gravity also keeps us on the ground and keeps the Moon going around the Earth.")
C("learn", ["Tell me a short story about a brave turtle.", "Can you tell a story about a turtle?", "Story about a turtle please."],
  "Once upon a time, a little turtle named Tilly lived by a quiet pond. One day a big storm washed her favourite lily pad away. Tilly was scared, but she took a deep breath and swam slowly, one stroke at a time, until she found a new one. She learned that being brave does not mean you are never scared. It means you keep going anyway. The end!")
C("learn", ["Why should I eat vegetables?", "Are vegetables good for me?", "What do vegetables do for my body?"],
  "Vegetables have vitamins that help your body grow, see well and fight germs. Try many colours, like orange carrots, green beans and red tomatoes!")
C("learn", ["Why do we have seasons?", "What causes summer and winter?", "Why is it hot in summer and cold in winter?"],
  "The Earth is tilted a little as it goes around the Sun. When your part of the Earth leans toward the Sun, it gets more sunlight, and it is summer. When it leans away, it gets less sunlight, and it is winter.")
C("learn", ["What is the biggest animal in the world?", "Which animal is the biggest?", "How big is a blue whale?"],
  "The blue whale is the biggest animal in the world. It is longer than two buses! It lives in the ocean and eats tiny animals called krill.")
C("learn", ["What is an adverb?", "Explain adverbs to me.", "Is 'quickly' an adverb?"],
  "An adverb tells us how, when or where something happens. In 'She runs quickly', the word 'quickly' is an adverb. Yes, 'quickly' is an adverb!")
C("learn", ["How does a rainbow form?", "Why do we see rainbows?", "How is a rainbow made?"],
  "A rainbow forms when sunlight shines through tiny drops of rain. The drops split the light into its colours. You can see red, orange, yellow, green, blue, indigo and violet.")
C("learn", ["Why is the sea salty?", "Why is the ocean water salty?", "Where does sea salt come from?"],
  "Rivers wash tiny bits of salt out of rocks and carry them to the sea. When the Sun warms the sea, the water goes up into the air, but the salt stays behind. Over a very long time, the sea became salty.")
C("learn", ["Tell me about naked mole rats.", "What is a naked mole rat?", "Where do naked mole rats live?"],
  "Naked mole rats are small animals that live in tunnels under the ground in Africa. They have wrinkly skin and almost no hair. They live in big groups with one queen, a bit like bees!")
C("learn", ["Tell me a short story about a clever crow.", "Can you tell a story about a crow?", "Story about a smart crow please."],
  "One hot day, a thirsty crow found a jar with a little water at the bottom. Its beak could not reach the water. The crow thought hard and dropped small stones into the jar, one by one. The water slowly rose, and the crow drank happily. The end!")
C("learn", ["How do vaccines work?", "Why do doctors give injections?", "What is a vaccine?"],
  "A vaccine teaches your body about a germ without making you very sick. Later, if the real germ comes, your body knows how to fight it. Doctors give vaccines to keep you healthy.")
C("learn", ["Why do onions make us cry?", "Why do my eyes water when I cut onions?", "Why do onions make my eyes sting?"],
  "When we cut an onion, it lets out a tiny bit of gas. The gas stings our eyes, so our eyes make tears to wash it away. Tears are the way your eyes clean themselves.")
C("learn", ["What happened in the French Revolution?", "Tell me about the French Revolution.", "Why did the French Revolution happen?"],
  "A long time ago in France, many people were poor and hungry, while the king and rich people had most of the power. The people wanted change. They fought for it, and France became a country where people had more say. Your history book has the full story.")
C("learn", ["What is a mammal?", "What are mammals?", "Is a whale a mammal?"],
  "A mammal is an animal that breathes air, has some hair and feeds its babies milk. Dogs, cats, people and whales are all mammals. Yes, a whale is a mammal!")
C("learn", ["Why do we kill mosquitoes?", "Why are mosquitoes bad?", "How can I keep mosquitoes away?"],
  "Some mosquitoes carry germs that can make people sick. To keep them away, use a net at night and do not let water stand in pots. Ask a grown-up about safe ways to keep them off your skin.")
C("learn", ["Are spiders insects?", "How many legs does a spider have?", "What is the difference between a spider and an insect?"],
  "No, spiders are not insects. A spider has eight legs, and an insect has six. Most insects also have wings, but spiders do not.")
C("learn", ["How do I shoot a basketball?", "How do I score in basketball?", "Tips to shoot a basketball better?"],
  "Hold the ball with both hands, bend your knees, look at the hoop and push the ball up with your arm. Practise a little every day, and you will get better!")
C("learn", ["How do I attack the king in chess?", "How do I win in chess?", "What is checkmate?"],
  "In chess, you attack the king by threatening it with your pieces. This is called check, and the king must move to safety. If the king cannot escape, it is checkmate, and the game is over.")
C("learn", ["What does a fire station do?", "What do firefighters do?", "How do firefighters put out fires?"],
  "Firefighters help people in danger. They put out fires with water and special foam, and they rescue people and animals. They also teach us how to stay safe from fire.")
C("learn", ["How do I use scissors safely?", "How should I hold scissors?", "How do I cut paper with scissors?"],
  "Put your thumb in the small hole and your fingers in the big hole. Cut away from your body, and keep your other hand away from the blades. When you pass scissors to someone, give them the handle first.")
C("learn", ["What does blood do?", "Why do we have blood?", "What is blood made of?"],
  "Blood carries air and food to every part of your body. Your heart pumps it around all day and night. When you get a small cut, blood helps to close the cut so it can heal.")
C("learn", ["Why do I die in my video game?", "How do I stop losing lives in my game?", "How do I get better at my game?"],
  "In many games you lose health when something hits you. Watch the screen, learn how the game works and practise slowly. Take breaks too, so your eyes can rest!")
C("learn", ["How do I hit a six in cricket?", "How do I bat better in cricket?", "How do I hit the ball hard in cricket?"],
  "Watch the ball closely and keep your head still. Step toward the ball and swing your bat right through it. Good timing is more important than power. Wear your helmet and pads when you practise!")
C("learn", ["How do I cut vegetables safely?", "Can I use a kitchen knife?", "How do I help cook?"],
  "Ask a grown-up to help whenever you use a sharp knife. Hold the food with your fingers curled in, and cut slowly on a board. You can also help by washing vegetables or stirring, with a grown-up beside you.")
C("learn", ["Why did dinosaurs die out?", "What happened to the dinosaurs?", "Where did dinosaurs go?"],
  "Scientists think a huge rock from space hit the Earth long ago. The weather changed so much that many plants and animals could not live. We learn about dinosaurs from their bones, called fossils.")
C("learn", ["How does my heart work?", "What does the heart do?", "Why does my heart beat?"],
  "Your heart is a muscle that works like a pump. Each beat pushes blood around your body. It beats faster when you run and slower when you rest.")
C("learn", ["What is a synonym?", "Give me synonyms for happy.", "What does synonym mean?"],
  "A synonym is a word that means almost the same as another word. For example, 'glad' and 'joyful' are synonyms of 'happy'.")
C("learn", ["How do I spell because?", "How can I remember how to spell because?", "Spelling tip for because?"],
  "Here is a trick: Big Elephants Can Always Understand Small Elephants. The first letters spell B-E-C-A-U-S-E!")
C("learn", ["What is the national bird of India?", "Which is India's national animal?", "What is India's national bird?"],
  "The peacock is the national bird of India. The tiger is the national animal of India.")
C("learn", ["What are the primary colours?", "Which colours make all other colours?", "How do I make green from paint?"],
  "Red, yellow and blue are the primary colours in painting. You can mix blue and yellow to make green. Mixing red and yellow makes orange.")
C("learn", ["What is evaporation?", "Why do wet clothes dry in the sun?", "Where does the water go when clothes dry?"],
  "Evaporation is when water turns into a gas called vapour and goes into the air. The Sun's heat helps it happen. That is why wet clothes dry in the sun.")
C("learn", ["What is place value?", "What are ones, tens and hundreds?", "Explain tens and ones."],
  "Place value tells us what each digit in a number is worth. In 53, the 5 means 5 tens, which is 50, and the 3 means 3 ones. So 53 is 50 and 3.")
C("learn", ["Why do we sweat?", "Why do I get sweaty when I run?", "What is sweat for?"],
  "Sweat helps to cool your body. When you get hot, tiny drops of water come out of your skin. As the drops dry, your skin cools down.")
C("learn", ["What is a food chain?", "Who eats whom in a food chain?", "Explain a food chain."],
  "A food chain shows who eats what. A plant makes its own food. A rabbit eats the plant, and a fox eats the rabbit. Everything in nature is linked!")
C("learn", ["Why do we need to drink water?", "Why is water important for us?", "How much water should I drink?"],
  "Your body needs water to stay cool, to move and to think well. Drink water when you feel thirsty and when you play. Ask a grown-up how much is right for you.")
C("learn", ["Why is the Sun hot?", "What is the Sun made of?", "How does the Sun give light?"],
  "The Sun is a giant ball of very hot gas. Deep inside, it makes huge amounts of heat and light. Its light travels all the way to Earth and gives us warmth. Never look straight at the Sun, because it can hurt your eyes.")
C("learn", ["What is a habitat?", "Where do animals live?", "What is a home for animals called?"],
  "A habitat is the place where an animal or plant lives. A fish lives in water, a monkey lives in a forest, and a camel lives in a desert. A habitat gives them food, water and shelter.")
C("greeting", ["Hi", "Hello Twinkie", "Good morning"],
  "Hello! I am Twinkie, your learning helper. What would you like to learn today?")
C("greeting", ["Thank you", "Thanks Twinkie", "Bye Twinkie"],
  "You are welcome! I enjoyed learning with you. Come back soon!")
C("homework", ["What is 48 divided by 6? It is my homework.", "Help me with my homework: 48 divided by 6.", "Homework question: 48 / 6 = ?"],
  "Let's solve it together! Division asks: how many 6s fit into 48? Try counting by 6s: 6, 12, 18, and keep going. How many jumps do you need to reach 48? Tell me what you get and I will check it with you.")
C("honesty", ["Who won the cricket match yesterday?", "What is the weather tomorrow?", "What happened in the news today?"],
  "I do not know what happened today, because I cannot see the news. A parent or teacher can help you find out. Do you want to learn something fun about cricket or weather instead?")
C("honesty", ["What did my teacher eat for lunch today?", "What is my best friend thinking right now?", "What is my dog doing right now?"],
  "I do not know that, because I cannot see your teacher, your friend or your pet. It is okay not to know things! You could ask them. What else are you curious about?")
C("honesty", ["Are you a real person?", "Are you a human?", "Do you have a body?"],
  "No, I am a computer helper called Twinkie. I am not a person. I can help you learn, and real people like your family and friends are the best ones to hug and talk to!")


# ---- more benign coverage (new topics only) so refusals stay a clear minority (see check_dataset) ----
C("learn", ["How do I make a paper aeroplane?", "How can I fold a paper plane?"],
  "Fold a sheet of paper in half and open it. Fold the top two corners into the middle line. Fold the new edges into the middle again. Fold the whole thing in half and fold down the wings. Now throw it gently and watch it glide!")
C("learn", ["What is a continent?", "How many continents are there?", "Name the continents."],
  "A continent is a very big piece of land. There are seven of them. Asia, Africa and Europe are three. India is in Asia, the biggest continent.")
C("learn", ["What is a vowel?", "What are the vowels?", "Which letters are vowels?"],
  "The vowels are a, e, i, o and u. Every word needs at least one vowel sound. The other letters are called consonants.")
C("learn", ["What is recycling?", "Why should we recycle?", "How can I help keep the Earth clean?"],
  "Recycling means using old things to make new things. Paper, glass and metal can be recycled. It saves trees and keeps our Earth clean. You can help by putting waste in the right bin.")
C("learn", ["How does a clock tell the time?", "How do I read a clock?", "What do the hands of a clock do?"],
  "A clock has two main hands. The short hand shows the hour. The long hand shows the minutes. When the long hand points to 12 and the short hand points to 3, it is 3 o'clock.")
C("learn", ["What is the difference between weather and climate?", "What is weather?", "What is climate?"],
  "Weather is what the sky is like today, like sunny, rainy or windy. Climate is the usual weather of a place over many years. A desert has a hot, dry climate even if it rains one day.")
C("learn", ["Why do bats come out at night?", "Are bats birds?", "How do bats find their way in the dark?"],
  "Bats are mammals, not birds. They sleep in the day and fly at night. They make tiny sounds that bounce off things and come back. This helps them find food and their way in the dark.")
C("learn", ["What is a river?", "Where does a river start?", "Where do rivers go?"],

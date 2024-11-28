from collections import defaultdict

TEMPLATE_PROMPT = "Adjust the colors of this scene to be {term}."

CONCEPTUAL_PROMPTS = [
    "Sadder",
	"Happier",
	"Dramatic",
	"Joyful",
	"Dreamy",
]

COLOR_PROMPTS = [
"Shades of red",
"Tints of blue",
"A purple palette",
"A green color scheme",
"A yellow hue",
]

PROMPTS = defaultdict(list)
for cp in CONCEPTUAL_PROMPTS:
    PROMPTS["conceptual"].append(TEMPLATE_PROMPT.format(term=cp.lower()))

for cp in COLOR_PROMPTS:
    PROMPTS["color"].append(TEMPLATE_PROMPT.format(term=cp.lower()))


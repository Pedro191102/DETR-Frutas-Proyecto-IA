"""Etiquetas oficiales LVIS que se usan como objetivos del detector de alimentos."""
from pathlib import Path

BASE = str(Path(__file__).resolve().parents[1] / "models" / "base")
CLASES = tuple("""
orange_(fruit) mandarin_orange clementine lemon lime strawberry raspberry blackberry
blueberry grape apple pear peach apricot cherry fig_(fruit) persimmon banana avocado
pineapple papaya kiwi_fruit coconut watermelon melon cantaloup date_(fruit) prune
tomato cucumber eggplant bell_pepper chili_(vegetable) zucchini pumpkin gourd pepper
lettuce celery asparagus artichoke broccoli cauliflower brussels_sprouts carrot potato
sweet_potato radish turnip onion green_onion garlic ginger green_bean pea_(food)
chickpea legume edible_corn
""".split())

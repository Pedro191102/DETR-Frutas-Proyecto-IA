from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    label: str
    backend: str
    repo_id: str
    classes: int | None
    dataset: str
    license: str
    description: str
    prompt_classes: tuple[str, ...] = ()


# Broad text vocabulary for YOLOE. These are inference-time prompts, not a claim
# that the checkpoint was fine-tuned on this exact list.
YOLOE_FRUIT_VEGETABLE_CLASSES: tuple[str, ...] = (
    # Fruits
    "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry",
    "Cantaloupe", "Cherry", "Coconut", "Cranberry", "Date", "Dragon fruit",
    "Durian", "Fig", "Grape", "Grapefruit", "Guava", "Jackfruit", "Kiwi",
    "Lemon", "Lime", "Longan", "Lychee", "Mandarin", "Mango", "Mangosteen",
    "Mulberry", "Nectarine", "Olive", "Orange", "Papaya", "Passion fruit",
    "Peach", "Pear", "Persimmon", "Pineapple", "Plantain", "Plum",
    "Pomegranate", "Pomelo", "Quince", "Rambutan", "Raspberry", "Star fruit",
    "Strawberry", "Tangerine", "Watermelon", "Honeydew melon", "Breadfruit",
    "Soursop", "Prickly pear", "Gooseberry", "Currant",
    # Vegetables, roots, legumes and edible fungi
    "Artichoke", "Arugula", "Asparagus", "Beetroot", "Bell pepper",
    "Bitter melon", "Bok choy", "Broccoli", "Brussels sprouts", "Cabbage",
    "Carrot", "Cassava", "Cauliflower", "Celery", "Chayote", "Chili pepper",
    "Collard greens", "Corn", "Cucumber", "Eggplant", "Fennel", "Garlic",
    "Ginger", "Green bean", "Green pea", "Kale", "Leek", "Lettuce",
    "Mushroom", "Okra", "Onion", "Parsnip", "Potato", "Pumpkin", "Radish",
    "Spinach", "Sweet potato", "Swiss chard", "Taro root", "Tomato", "Turnip",
    "Yam", "Zucchini", "Butternut squash", "Napa cabbage", "Scallion",
    "Shallot", "Turmeric", "Watercress", "Jalapeno pepper", "Serrano pepper",
    "Pea pod", "Red cabbage", "Bean sprout", "Bamboo shoot", "Lotus root",
    "Moringa leaf", "Green chili",
)


DETECTORS: dict[str, DetectorSpec] = {
    "yoloe_fruitveg_open": DetectorSpec(
        key="yoloe_fruitveg_open",
        label=f"YOLOE-26s · vocabulario abierto · {len(YOLOE_FRUIT_VEGETABLE_CLASSES)} categorías",
        backend="yoloe",
        repo_id="yoloe-26s-seg.pt",
        classes=len(YOLOE_FRUIT_VEGETABLE_CLASSES),
        dataset="Modelo abierto de Ultralytics; categorías indicadas mediante prompts de texto.",
        license="AGPL-3.0 (Ultralytics)",
        description=(
            f"Detector de vocabulario abierto. Busca {len(YOLOE_FRUIT_VEGETABLE_CLASSES)} nombres "
            "de frutas, verduras, raíces y hongos mediante prompts de texto. No fue ajustado "
            "específicamente con nuestro dataset ni garantiza la misma precisión en todas las clases."
        ),
        prompt_classes=YOLOE_FRUIT_VEGETABLE_CLASSES,
    ),
}

from .lvis_clases import BASE, CLASES

ROOT = Path(__file__).resolve().parents[1]
DETECTORS["detr_lvis"] = DetectorSpec("detr_lvis", "DETR",
    "detr", BASE, len(CLASES), "LVIS v1", "Apache-2.0",
    "Modelo base incluido. Muestra las 58 etiquetas de alimentos seleccionadas.")
puntero = ROOT / "models" / "detr_actual.txt"
if puntero.is_file():
    ruta = ROOT / puntero.read_text().strip()
    if (ruta / "config.json").is_file():
        DETECTORS["detr_propio"] = DetectorSpec("detr_propio", "DETR 2",
            "detr", str(ruta), len(CLASES), "Subconjunto de LVIS v1", "Apache-2.0",
            "Mejor época según la pérdida de validación de la última corrida finalizada.")

CUSTOM_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "ajustado.pt"
if CUSTOM_WEIGHTS.is_file():
    DETECTORS["yoloe_ajustado"] = DetectorSpec(
        key="yoloe_ajustado", label="YOLOE-26s · ajuste propio", backend="yoloe",
        repo_id="models/ajustado.pt", classes=None, dataset="Dataset del entrenamiento.",
        license="AGPL-3.0 (Ultralytics)", description="Usa las clases guardadas en el checkpoint.",
    )


def get_detector_spec(key: str) -> DetectorSpec:
    return DETECTORS[key]

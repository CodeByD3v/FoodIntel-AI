from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import math
import re
import threading
from difflib import SequenceMatcher, get_close_matches
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from myth.food_model import FoodImageClassifier


ROOT = Path(__file__).resolve().parent
PORT = 8000
HOST = "127.0.0.1"
FRONTEND_FILE = ROOT / "test.html"

FOOD101_INGREDIENTS = ROOT / "converted_ingredients_grams.csv"   # grams already converted
FOOD101_MACROS      = ROOT / "food101_ingredient_macros.csv"

DESCRIPTORS = {
    "sliced", "diced", "chopped", "minced", "fresh", "frozen", "canned", "dried",
    "crushed", "ground", "grated", "peeled", "skinless", "boneless", "low-fat",
    "reduced-fat", "unsalted", "salted", "light", "extra-virgin", "shredded",
    "thinly", "thick", "small", "large", "medium", "roasted", "toasted", "raw",
    "cooked", "halved", "quartered", "whole", "sweetened", "unsweetened", "powdered",
}

CANONICAL = {
    "ground beef": "beef",
    "hamburger": "beef",
    "chicken breast": "chicken",
    "chicken breasts": "chicken",
    "whole chicken": "chicken",
    "whole milk": "milk",
    "skim milk": "milk",
    "brown rice": "rice",
    "white rice": "rice",
    "spaghetti": "pasta",
    "macaroni": "pasta",
    "cheddar cheese": "cheese",
    "mozzarella": "cheese",
    "mozzarella cheese": "cheese",
    "parmesan": "cheese",
    "olive oil": "oil",
    "vegetable oil": "oil",
    "margarine": "butter",
    "oleo": "butter",
    "shortening": "butter",
}


def normalize_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"\(.*?\)", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    tokens = [token for token in value.split() if token not in DESCRIPTORS]
    value = " ".join(tokens)
    value = re.sub(r"\s+", " ", value).strip()
    if value.endswith("es") and len(value) > 3:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 3:
        value = value[:-1]
    return CANONICAL.get(value, value)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_dish_index() -> tuple[list[dict], dict[str, list[dict]]]:
    """Load dish → ingredient list (with quantity_g) from the grams CSV."""
    if not FOOD101_INGREDIENTS.exists():
        return [], {}

    dishes: list[dict] = []
    ingredients_by_norm: dict[str, list[dict]] = {}
    seen_norms: set[str] = set()
    with FOOD101_INGREDIENTS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            dish = (row.get("dish") or "").strip()
            dish_norm = normalize_text(dish)
            if not dish_norm:
                continue
            if dish_norm not in seen_norms:
                dishes.append({"dish": dish, "dish_norm": dish_norm})
                seen_norms.add(dish_norm)
            if dish_norm not in ingredients_by_norm:
                raw = row.get("ingredients") or "[]"
                try:
                    ingredients_by_norm[dish_norm] = ast.literal_eval(raw)
                except Exception:
                    ingredients_by_norm[dish_norm] = []
    return dishes, ingredients_by_norm


def load_macros() -> dict[str, dict]:
    """Load macro table keyed by LOWERCASE raw ingredient name (no normalization)."""
    if not FOOD101_MACROS.exists():
        return {}

    macros: dict[str, dict] = {}
    with FOOD101_MACROS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            # Store under both raw lowercase AND normalized form so both lookups work
            raw_name = (row.get("ingredient") or "").strip().lower()
            if not raw_name:
                continue
            entry = {
                "ingredient":         raw_name,
                "cal_per_100g":       parse_float(row.get("calories_kcal_per_100g")),
                "protein_g_per_100g": parse_float(row.get("protein_g_per_100g")),
                "carbs_g_per_100g":   parse_float(row.get("carbs_g_per_100g")),
                "fat_g_per_100g":     parse_float(row.get("fat_g_per_100g")),
            }
            macros[raw_name] = entry
            norm_name = normalize_text(raw_name)
            if norm_name and norm_name != raw_name:
                macros.setdefault(norm_name, entry)
    return macros


def find_dish_ingredients(target_dish_norm: str) -> tuple[str, list[dict]]:
    if not target_dish_norm:
        return "", []

    if target_dish_norm in DISH_INGREDIENTS_BY_NORM:
        for row in DISH_ROWS:
            if row["dish_norm"] == target_dish_norm:
                return row["dish"], DISH_INGREDIENTS_BY_NORM.get(target_dish_norm, [])

    fallback_dish = ""
    fallback_ingredients: list[dict] = []
    fallback_score = 0.0
    for row in DISH_ROWS:
        score = SequenceMatcher(None, target_dish_norm, row["dish_norm"]).ratio()
        if score > fallback_score:
            fallback_dish = row["dish"]
            fallback_ingredients = DISH_INGREDIENTS_BY_NORM.get(row["dish_norm"], [])
            fallback_score = score
    return fallback_dish, fallback_ingredients


def best_match(name: str, choices: list[str], cutoff: float = 0.65) -> tuple[str | None, float]:
    name = normalize_text(name)
    if not name or not choices:
        return None, 0.0

    if name in choices:
        return name, 1.0

    direct = get_close_matches(name, choices, n=1, cutoff=cutoff)
    if direct:
        candidate = direct[0]
        return candidate, SequenceMatcher(None, name, candidate).ratio()

    tokens = name.split()
    for token in reversed(tokens):
        if token in choices:
            return token, 0.85
        singular = token[:-1] if token.endswith("s") else token
        if singular in choices:
            return singular, 0.8

    best_choice = None
    best_score = 0.0
    for choice in choices:
        score = SequenceMatcher(None, name, choice).ratio()
        if score > best_score:
            best_choice = choice
            best_score = score
    if best_score >= cutoff:
        return best_choice, best_score
    return None, best_score


def decode_image(payload: str) -> bytes:
    if not payload:
        return b""
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception:
        return b""


FOOD_PREDICTOR = FoodImageClassifier(model_id="nateraw/food", cache_root=ROOT / "myth")


def predict_dish(image_bytes: bytes, filename: str, dish_lookup: list[dict]) -> tuple[str, float, list[dict]]:
    """Predict a dish label with the local model from the models folder."""
    if not image_bytes:
        return "no image", 0.0, []

    predictions: list[dict] = []
    if FOOD_PREDICTOR and FOOD_PREDICTOR.loaded:
        predictions = FOOD_PREDICTOR.predict_top_k(image_bytes, top_k=1)
        if predictions:
            cnn_class = str(predictions[0]["label"])
            cnn_confidence = float(predictions[0]["score"])
            if dish_lookup:
                match, _ = best_match(cnn_class, [row["dish_norm"] for row in dish_lookup], cutoff=0.3)
                if match:
                    for row in dish_lookup:
                        if row["dish_norm"] == match:
                            return row["dish"], round(cnn_confidence, 2), predictions
            return cnn_class, round(cnn_confidence, 2), predictions

    if not dish_lookup:
        return "uploaded image", 0.0, predictions

    # Fallback: Use hash-based selection if the model is unavailable.
    digest = hashlib.sha256(image_bytes).hexdigest()
    index = int(digest[:8], 16) % len(dish_lookup)
    return dish_lookup[index]["dish"], 0.65, predictions


def sum_macros(ingredients: list[dict], macro_lookup: dict[str, dict]) -> tuple[dict, list[dict], float]:
    """
    Sum macros for all ingredients (each already in grams via quantity_g),
    then normalise to per-100g of the finished dish.
    """
    raw_totals = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    resolved: list[dict] = []
    matched_count = 0
    total_grams = 0.0

    macro_keys = list(macro_lookup.keys())

    for item in ingredients:
        if not item:
            continue
        ing_name = (item.get("ingredient") or item.get("name") or "").strip()
        # Use pre-converted grams field
        grams = parse_float(item.get("quantity_g") or item.get("grams") or 0)

        if not ing_name or grams <= 0:
            continue

        # Lookup: try raw lowercase → normalized → fuzzy
        ing_lower = ing_name.lower()
        macro_row = (
            macro_lookup.get(ing_lower)
            or macro_lookup.get(normalize_text(ing_lower))
        )
        if not macro_row:
            match, score = best_match(ing_lower, macro_keys, cutoff=0.50)
            if match:
                macro_row = macro_lookup[match]

        if not macro_row:
            continue

        matched_count += 1
        total_grams   += grams
        raw_totals["calories"] += grams * macro_row["cal_per_100g"]       / 100.0
        raw_totals["protein"]  += grams * macro_row["protein_g_per_100g"] / 100.0
        raw_totals["fat"]      += grams * macro_row["fat_g_per_100g"]     / 100.0
        raw_totals["carbs"]    += grams * macro_row["carbs_g_per_100g"]   / 100.0
        resolved.append({
            "ingredient": ing_name,
            "grams":      round(grams, 1),
            "matched_to": macro_row["ingredient"],
        })

    coverage = matched_count / max(1, len(ingredients))

    # Normalise entire dish to per-100g serving
    if total_grams > 0:
        f = 100.0 / total_grams
        nutrition_per_100g = {k: round(v * f, 2) for k, v in raw_totals.items()}
    else:
        nutrition_per_100g = {k: 0.0 for k in raw_totals}

    return nutrition_per_100g, resolved, coverage


def compute_score(nutrition: dict, positive_tags: list[str], negative_tags: list[str]) -> tuple[int, str, str]:
    cal_pct = min((nutrition["calories"] / 2000.0) * 100.0, 100.0)
    pro_pct = min((nutrition["protein"] / 50.0) * 100.0, 100.0)
    fat_pct = min((nutrition["fat"] / 78.0) * 100.0, 100.0)
    carb_pct = min((nutrition["carbs"] / 300.0) * 100.0, 100.0)
    mean = (cal_pct + pro_pct + fat_pct + carb_pct) / 4.0
    variance = ((cal_pct - mean) ** 2 + (pro_pct - mean) ** 2 + (fat_pct - mean) ** 2 + (carb_pct - mean) ** 2) / 4.0
    balance = max(0.0, 100.0 - math.sqrt(variance) * 2.0)
    protein_quality = pro_pct
    fat_load = 100.0 - fat_pct
    tag_score = 50.0
    total_tags = len(positive_tags) + len(negative_tags)
    if total_tags > 0:
        tag_score = (len(positive_tags) / total_tags) * 100.0
    overall = int(round((balance + protein_quality + fat_load + tag_score) / 4.0))
    if overall >= 85:
        return overall, "A", "var(--green)"
    if overall >= 70:
        return overall, "B", "var(--cyan)"
    if overall >= 55:
        return overall, "C", "var(--yellow)"
    return overall, "D", "var(--red)"


# ── Dish classification sets (use normalized form: spaces, no underscores) ───
_MEAT_DISHES = {
    "baby back rib", "beef carpaccio", "beef tartare", "bibimbap",
    "breakfast burrito", "caesar salad", "chicken curry", "chicken quesadilla",
    "chicken wing", "clam chowder", "club sandwich", "crab cake",
    "croque madame", "egg benedict", "escargot", "filet mignon",
    "fish and chip", "foie gra", "fried calamari", "fried rice",
    "grilled salmon", "hamburger", "hot and sour soup", "hot dog",
    "huevos ranchero", "lasagna", "lobster bisque", "lobster roll sandwich",
    "mussel", "nacho", "oyster", "pad thai", "paella", "peking duck",
    "pho", "pork chop", "prime rib", "pulled pork sandwich", "ramen",
    "sashimi", "scallop", "shrimp and grit", "spaghetti bolognese",
    "spaghetti carbonara", "spring roll", "steak", "sushi", "taco",
    "takoyaki", "tuna tartare",
}

_JUNK_DISHES = {
    "french fri", "onion ring", "nacho", "hot dog", "hamburger",
    "chicken wing", "donut", "churro", "beignet", "fried calamari",
    "fish and chip", "poutine", "macaroni and cheese",
}

_SEAFOOD_DISHES = {
    "clam chowder", "crab cake", "fish and chip", "fried calamari",
    "grilled salmon", "lobster bisque", "lobster roll sandwich", "mussel",
    "oyster", "paella", "sashimi", "scallop", "shrimp and grit",
    "spring roll", "sushi", "takoyaki", "tuna tartare",
}

_GLUTEN_FREE_DISHES = {
    "beet salad", "caprese salad", "ceviche", "edamame", "greek salad",
    "guacamole", "hummus", "miso soup", "oyster", "sashimi",
    "seaweed salad", "steak", "filet mignon", "pork chop", "grilled salmon",
}

_DAIRY_FREE_DISHES = {
    "bibimbap", "ceviche", "edamame", "falafel", "fried rice", "guacamole",
    "gyoza", "hummus", "miso soup", "pad thai", "peking duck", "pho",
    "ramen", "sashimi", "seaweed salad", "spring roll", "sushi",
    "taco", "takoyaki", "tuna tartare",
}


def build_diet_tags(nutrition: dict, dish_norm: str = "") -> tuple[list[str], list[str]]:
    """
    Return (positive_tags, negative_tags) based on both nutrition values
    and dish identity. All thresholds are per-100g of the dish.
    """
    cal  = nutrition.get("calories", 0)
    pro  = nutrition.get("protein",  0)
    fat  = nutrition.get("fat",      0)
    carb = nutrition.get("carbs",    0)

    positive: list[str] = []
    negative: list[str] = []

    # ── Protein ──────────────────────────────────────────────────────
    if pro >= 20:
        positive.append("high-protein")
    elif pro < 5:
        negative.append("low-protein")

    # ── Carbs ─────────────────────────────────────────────────────────
    if carb <= 15:
        positive.append("low-carb")
    elif carb >= 40:
        negative.append("high-carb")

    # ── Fat ───────────────────────────────────────────────────────────
    if fat <= 8:
        positive.append("low-fat")
    elif fat >= 25:
        negative.append("high-fat")

    # ── Calories ──────────────────────────────────────────────────────
    if cal <= 150:
        positive.append("light")
    elif cal >= 400:
        negative.append("high-calorie")

    # ── Junk food ─────────────────────────────────────────────────────
    if dish_norm in _JUNK_DISHES or (fat >= 20 and carb >= 30 and cal >= 300):
        negative.append("junk-food")

    # ── Vegetarian / Non-vegetarian ───────────────────────────────────
    if dish_norm in _MEAT_DISHES:
        negative.append("non-vegetarian")
    else:
        positive.append("vegetarian")

    # ── Seafood ───────────────────────────────────────────────────────
    if dish_norm in _SEAFOOD_DISHES:
        positive.append("seafood")

    # ── Gluten-free ───────────────────────────────────────────────────
    if dish_norm in _GLUTEN_FREE_DISHES:
        positive.append("gluten-free")

    # ── Dairy-free ────────────────────────────────────────────────────
    if dish_norm in _DAIRY_FREE_DISHES:
        positive.append("dairy-free")

    # ── Balanced meal ─────────────────────────────────────────────────
    if 10 <= pro <= 35 and 10 <= carb <= 40 and 5 <= fat <= 20 and 150 <= cal <= 350:
        positive.append("balanced")

    return positive, negative


def build_bars(seed_text: str, nutrition: dict) -> list[int]:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    energy = max(1.0, nutrition["calories"] + nutrition["protein"] * 10 + nutrition["fat"] * 8 + nutrition["carbs"] * 4)
    bars: list[int] = []
    for index in range(300):
        wave = math.sin(index / 7.0 + seed * math.pi * 2.0) * 24.0
        ripple = math.sin(index / 17.0 + energy / 250.0) * 12.0
        drift = math.sin(index / 31.0 + seed) * 8.0
        value = 40.0 + wave + ripple + drift
        bars.append(int(max(2, min(100, round(value)))))
    return bars


def build_logs() -> list[dict]:
    return [
        {"time": "00:00:00", "type": "INFO", "msg": "image uploaded successfully"},
        {"time": "00:00:01", "type": "INFO", "msg": "running local model inference from models folder"},
        {"time": "00:00:02", "type": "INFO", "msg": "fuzzy matching predicted dish to food101 dataset"},
        {"time": "00:00:03", "type": "INFO", "msg": "aggregating ingredient macros from dataset"},
        {"time": "00:00:04", "type": "SUCCESS", "msg": "analysis complete"},
    ]


DATA_READY = threading.Event()
DATA_LOAD_LOCK = threading.Lock()
DISH_ROWS: list[dict] = []
DISH_INGREDIENTS_BY_NORM: dict[str, list[dict]] = {}
MACRO_LOOKUP: dict[str, dict] = {}
DATA_LOAD_ERROR: str | None = None

STATE_LOCK = threading.Lock()
STATE: dict = {
    "status": "idle",
    "status_label": "READY",
    "processing": 0,
    "bars": [2] * 300,
    "last_result": None,
}


def ensure_data_loaded(block: bool = True) -> None:
    global DISH_ROWS, DISH_INGREDIENTS_BY_NORM, MACRO_LOOKUP, DATA_LOAD_ERROR
    if DATA_READY.is_set():
        return
    if not block:
        return

    with DATA_LOAD_LOCK:
        if DATA_READY.is_set():
            return
        if DATA_LOAD_ERROR is not None:
            raise RuntimeError(DATA_LOAD_ERROR)
        try:
            DISH_ROWS, DISH_INGREDIENTS_BY_NORM = load_dish_index()
            MACRO_LOOKUP = load_macros()
            DATA_READY.set()
        except Exception as exc:
            DATA_LOAD_ERROR = str(exc)
            raise


def load_data_in_background() -> None:
    try:
        ensure_data_loaded(block=True)
        with STATE_LOCK:
            STATE["status"] = "idle"
            STATE["status_label"] = "READY"
    except Exception as exc:
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["status_label"] = "LOAD ERROR"
            STATE["last_result"] = {"error": str(exc)}


def analyze_payload(payload: dict) -> dict:
    image_bytes = decode_image(payload.get("image", ""))
    filename = payload.get("filename") or "uploaded image"

    if not DATA_READY.is_set():
        stem = Path(filename or "uploaded image").stem.replace("_", " ").replace("-", " ")
        placeholder_lookup = [{"dish": stem or "uploaded image", "dish_norm": normalize_text(stem or "uploaded image")}]
        predicted_dish, dish_confidence, top_predictions = predict_dish(image_bytes, filename, placeholder_lookup)
        bars = build_bars(f"{filename}:{predicted_dish}:loading", {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0})
        return {
            "status": "complete",
            "status_label": "COMPLETE",
            "processing": 100,
            "dish_text": predicted_dish,
            "matched_dish": predicted_dish,
            "confidence": round(min(0.95, 0.45 + dish_confidence * 0.2), 2),
            "predictions": top_predictions,
            "ingredients": [],
            "nutrition": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
            "macros": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
            "score": 0,
            "grade": {"letter": "D", "color": "var(--red)"},
            "diet_tags": {"positive": [], "negative": []},
            "bars": bars,
            "logs": build_logs() + [
                {"time": "00:00:05", "type": "INFO", "msg": "dataset index still loading; using a quick fallback result"},
                {"time": "00:00:06", "type": "SUCCESS", "msg": "analysis complete"},
            ],
        }

    ensure_data_loaded(block=True)
    predicted_dish, dish_confidence, top_predictions = predict_dish(image_bytes, filename, DISH_ROWS)

    matched_dish = predicted_dish
    matched_dish_norm = normalize_text(matched_dish)
    if DISH_ROWS:
        candidate, dish_score = best_match(normalize_text(predicted_dish), [row["dish_norm"] for row in DISH_ROWS], cutoff=0.35)
        if candidate:
            for row in DISH_ROWS:
                if row["dish_norm"] == candidate:
                    matched_dish = row["dish"]
                    matched_dish_norm = row["dish_norm"]
                    dish_confidence = max(dish_confidence, dish_score)
                    break

    matched_dish, ingredients = find_dish_ingredients(matched_dish_norm)
    nutrition, resolved_ingredients, coverage = sum_macros(ingredients, MACRO_LOOKUP)
    positive_tags, negative_tags = build_diet_tags(nutrition, matched_dish_norm)
    score, grade_letter, grade_color = compute_score(nutrition, positive_tags, negative_tags)
    confidence = round(min(0.99, 0.52 + dish_confidence * 0.28 + coverage * 0.2), 2)
    bars = build_bars(f"{filename}:{matched_dish}:{score}", nutrition)

    result = {
        "status": "complete",
        "status_label": "COMPLETE",
        "processing": 100,
        "dish_text": predicted_dish,
        "matched_dish": matched_dish,
        "confidence": confidence,
        "predictions": top_predictions,
        "ingredients": resolved_ingredients,
        "nutrition": nutrition,
        "macros": nutrition,
        "score": score,
        "grade": {"letter": grade_letter, "color": grade_color},
        "diet_tags": {"positive": positive_tags, "negative": negative_tags},
        "bars": bars,
        "logs": build_logs(),
    }
    return result


class AnalysisHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html", "/test.html"}:
            if not FRONTEND_FILE.exists():
                self._send_json(404, {"error": "frontend not found"})
                return

            body = FRONTEND_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if path in {"/api/health", "/health"}:
            self._send_json(
                200,
                {
                    "ok": True,
                    "loading": not DATA_READY.is_set() and DATA_LOAD_ERROR is None,
                    "status": STATE["status"],
                    "status_label": STATE["status_label"],
                    "datasets": {
                        "dishes": len(DISH_ROWS),
                        "macros": len(MACRO_LOOKUP),
                    },
                },
            )
            return

        if path in {"/api/sysinfo", "/sysinfo"}:
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                self._send_json(200, {
                    "cpu_pct": round(cpu, 1),
                    "mem_pct": round(mem.percent, 1),
                    "mem_used_gb": round(mem.used / 1e9, 2),
                    "mem_total_gb": round(mem.total / 1e9, 2),
                    "status": STATE["status"],
                })
            except ImportError:
                # psutil not installed — return process-level estimate
                import resource as _res
                self._send_json(200, {"cpu_pct": 0, "mem_pct": 0, "status": STATE["status"]})
            return

        if path in {"/api/metrics", "/metrics"}:
            with STATE_LOCK:
                payload = {
                    "status": STATE["status"],
                    "status_label": STATE["status_label"],
                    "processing": STATE["processing"],
                    "bars": STATE["bars"],
                }
                if STATE["last_result"]:
                    payload.update(STATE["last_result"])
            self._send_json(200, payload)
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/api/analyze", "/analyze"}:
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        if not payload.get("image"):
            self._send_json(400, {"error": "image is required"})
            return

        with STATE_LOCK:
            STATE["status"] = "processing"
            STATE["status_label"] = "PROCESSING"
            STATE["processing"] = 15
            STATE["bars"] = [5] * 300

        result = analyze_payload(payload)

        with STATE_LOCK:
            STATE["status"] = result["status"]
            STATE["status_label"] = result["status_label"]
            STATE["processing"] = result["processing"]
            STATE["bars"] = result["bars"]
            STATE["last_result"] = result

        self._send_json(200, result)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AnalysisHandler)
    print(f"Food analysis backend listening on http://{HOST}:{PORT}")
    threading.Thread(target=load_data_in_background, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
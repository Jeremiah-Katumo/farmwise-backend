# Offline fallback
offline_faq = json.loads(Path("offline_faq.json").read_text())
category_prompts = {
    "crop": "You are an agricultural advisor.",
    "weather": "You are a weather forecaster.",
    "pest": "You are a plant health expert.",
    "market": "You are a farm market analyst.",
    "season": "You are a horticulture expert."
}
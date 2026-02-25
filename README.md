from pathlib import Path

descriptions = {
    "model_assets": "Stored neural network artifacts (weights, scalers).",
    "templates": "HTML templates for the web interface.",
    "static": "Static frontend resources (CSS, JS, images).",
    "app.py": "Main application entry point.",
    "inference.py": "Model inference and prediction logic.",
    "requirements.txt": "Project dependencies."
}

with open("README.md", "w", encoding="utf-8") as f:
    f.write("# neuro_influence\n\n")
    f.write("## Repository structure\n\n")
    for path in Path(".").iterdir():
        name = path.name
        if name in descriptions:
            f.write(f"- `{name}` – {descriptions[name]}\n")
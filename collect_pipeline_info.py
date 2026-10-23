from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pipeline_info.txt"

patterns = [
    r"model\.joblib",
    r"6919",
    r"final_features",
    r"feature_names",
    r"tfidf",
    r"vectorizer",
    r"scaler",
    r"\.transform\(",
    r"preprocess",
    r"ColumnTransformer",
    r"TfidfVectorizer",
    r"StandardScaler",
    r"OneHotEncoder",
    r"joblib\.load",
    r"pickle\.load",
]

extensions = {".py", ".json", ".yaml", ".yml", ".txt", ".md"}

lines = []

lines.append("=" * 80)
lines.append("SMARTSOC PRODUCTION PIPELINE DIAGNOSTIC")
lines.append("=" * 80)
lines.append(f"PROJECT ROOT: {ROOT}")
lines.append("")

# ------------------------------------------------------------------
# 1. Production model directory
# ------------------------------------------------------------------
lines.append("=" * 80)
lines.append("1. PRODUCTION MODEL ARTIFACTS")
lines.append("=" * 80)

model_dir = ROOT / "models" / "windows_apt_v4_1" / "production"

if model_dir.exists():
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            try:
                size = p.stat().st_size
            except Exception:
                size = "unknown"
            lines.append(f"{p.relative_to(ROOT)} | {size} bytes")
else:
    lines.append("Production model directory NOT FOUND.")

lines.append("")

# ------------------------------------------------------------------
# 2. Search source files
# ------------------------------------------------------------------
lines.append("=" * 80)
lines.append("2. SOURCE REFERENCES TO PRODUCTION PIPELINE")
lines.append("=" * 80)

matches = 0

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    if path.suffix.lower() not in extensions:
        continue

    # Skip virtual environments and caches
    parts_lower = {p.lower() for p in path.parts}

    if any(x in parts_lower for x in {
        ".venv",
        ".venv-compat",
        "__pycache__",
        "node_modules",
        ".git"
    }):
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    file_matches = []

    for lineno, line in enumerate(text.splitlines(), 1):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
            file_matches.append((lineno, line.rstrip()))

    if file_matches:
        matches += len(file_matches)
        lines.append("")
        lines.append(f"FILE: {path.relative_to(ROOT)}")

        for lineno, line in file_matches:
            lines.append(f"  {lineno}: {line}")

lines.append("")
lines.append(f"TOTAL MATCHING LINES: {matches}")
lines.append("")

# ------------------------------------------------------------------
# 3. Inspect likely inference/service files
# ------------------------------------------------------------------
lines.append("=" * 80)
lines.append("3. LIKELY INFERENCE / SERVICE FILES")
lines.append("=" * 80)

for path in sorted(ROOT.rglob("*.py")):
    name = path.name.lower()

    if any(word in name for word in [
        "inference",
        "service",
        "predict",
        "model",
        "feature",
        "pipeline"
    ]):
        if ".venv" in path.parts or ".venv-compat" in path.parts:
            continue

        lines.append(str(path.relative_to(ROOT)))

lines.append("")

# ------------------------------------------------------------------
# 4. Production inference service
# ------------------------------------------------------------------
service_candidates = [
    ROOT / "src" / "service" / "smartsoc_inference_service_v4_1.py",
    ROOT / "src" / "smartsoc_inference_service_v4_1.py",
]

lines.append("=" * 80)
lines.append("4. PRODUCTION INFERENCE SERVICE")
lines.append("=" * 80)

for service in service_candidates:
    if service.exists():
        lines.append(f"FOUND: {service.relative_to(ROOT)}")
        lines.append("")
        try:
            text = service.read_text(encoding="utf-8", errors="ignore")
            lines.append(text)
        except Exception as e:
            lines.append(f"Could not read file: {e}")
        break
else:
    lines.append("Known production inference service path was not found.")

lines.append("")

# ------------------------------------------------------------------
# 5. Feature dimensions / model metadata
# ------------------------------------------------------------------
lines.append("=" * 80)
lines.append("5. PRODUCTION METADATA")
lines.append("=" * 80)

metadata_names = [
    "manifest.json",
    "environment.json",
    "feature_names.json",
    "features.json",
    "thresholds.json",
    "labels.json",
]

if model_dir.exists():
    for name in metadata_names:
        path = model_dir / name

        if path.exists():
            lines.append("")
            lines.append(f"--- {path.relative_to(ROOT)} ---")

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                # Avoid accidentally producing enormous output.
                if len(text) > 30000:
                    text = text[:30000]
                    text += "\n...[TRUNCATED]..."

                lines.append(text)

            except Exception as e:
                lines.append(f"Could not read: {e}")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
OUT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("")
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Diagnostic file created:")
print(OUT)
print("")
print("Upload this file to ChatGPT.")
print("")
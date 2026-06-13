import json

source_path = "cgbench_filtered.json"
clean_path = "cgbench_filtered_clean.json"

def to_hhmmss(seconds):
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def transform(item):
    vid = item.get("video_id", "")
    if vid.startswith("cgbench_"):
        item["video_id"] = vid[len("cgbench_"):]
    for interval in item.get("evidence_intervals", []):
        if "start" in interval and isinstance(interval["start"], (int, float)):
            interval["start"] = to_hhmmss(interval["start"])
        if "end" in interval and isinstance(interval["end"], (int, float)):
            interval["end"] = to_hhmmss(interval["end"])
    return item

with open(source_path, "r", encoding="utf-8") as f:
    source = json.load(f)

with open(clean_path, "r", encoding="utf-8") as f:
    clean = json.load(f)

existing_ids = {item["video_id"] for item in clean}

new_items = []
for item in source:
    raw_id = item.get("video_id", "")
    normalized_id = raw_id[len("cgbench_"):] if raw_id.startswith("cgbench_") else raw_id
    if normalized_id not in existing_ids:
        new_items.append(transform(item))

if new_items:
    clean.extend(new_items)
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"Added {len(new_items)} new record(s). Total: {len(clean)}")
else:
    print("No new records found.")

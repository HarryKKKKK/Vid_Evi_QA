import json

input_path = "cgbench_filtered.json"
output_path = "cgbench_filtered_clean.json"

def to_hhmmss(seconds):
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    vid = item.get("video_id", "")
    if vid.startswith("cgbench_"):
        item["video_id"] = vid[len("cgbench_"):]

    for interval in item.get("evidence_intervals", []):
        if "start" in interval:
            interval["start"] = to_hhmmss(interval["start"])
        if "end" in interval:
            interval["end"] = to_hhmmss(interval["end"])

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. {len(data)} records written to {output_path}")

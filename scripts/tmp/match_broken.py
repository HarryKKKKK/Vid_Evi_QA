#!/usr/bin/env python3
"""
用 out_name() 原样逻辑，把 broken 文件名与 cgbench_filtered.json 的 entry 做精确匹配。
不对文件名做任何切分或猜测。
"""
import json

FILTERED_JSON = "cgbench_pipeline/cgbench_filtered.json"
SUBSET_JSON = "cgbench_pipeline/cgbench_filtered.broken_subset.json"
OUTPUT_EXT = ".mp4"

BROKEN_BASENAMES = {
    "2e1X8BD3yF8_q9371_freeze.mp4",
    "BV1Xh411W7WL_q45_freeze.mp4",
    "BV1yG411C7Xi_q66_freeze.mp4",
    "qNU5q5a97v8_q9705_freeze.mp4",
    "aTV3damvdZs_q12_freeze.mp4",
    "BV1w44y147ph_q9795_freeze.mp4",
    "BV1pM4m1174T_q8348_freeze.mp4",
    "zPuLGkluoTY_q9338_freeze.mp4",
    "HTS2plAOXDM_q8062_freeze.mp4",
    "BV1rj411j7mx_q74_freeze.mp4",
    "BV14C4y1d7UG_q9258_freeze.mp4",
    "BV17t421u7eZ_q9940_freeze.mp4",
    "4MK89zVlYdQ_q8493_freeze.mp4",
    "BV1jx4y187pS_q7849_freeze.mp4",
    "BV18B4y1n7kF_q8404_freeze.mp4",
    "BV1hSafe4EJs_q8606_freeze.mp4",
    "gED3SwLUEzs_q8481_freeze.mp4",
    "CngNITEpLuU_q8437_freeze.mp4",
    "HTS2plAOXDM_q8027_freeze.mp4",
    "BV1yj421S7NZ_q9199_freeze.mp4",
    "BV1Dm42137sk_q8883_freeze.mp4",
    "AYJAJWCS8eM_q9559_freeze.mp4",
    "BV1eU421d71N_q9176_freeze.mp4",
    "BV1Hd4y1T7wK_q8283_freeze.mp4",
    "VaOM8TslFqg_q8394_freeze.mp4",
    "v72EanuzqHc_q9132_freeze.mp4",
    "9GUVLNoRFJ4_q8512_freeze.mp4",
    "pF8rcouqqdI_q8596_freeze.mp4",
    "BeAur755crI_q8929_freeze.mp4",
    "5kozt0uDa4c_q8454_freeze.mp4",
    "pF8rcouqqdI_q7906_freeze.mp4",
    "HTS2plAOXDM_q9096_freeze.mp4",
    "W1X32VHJ7fM_q9141_freeze.mp4",
    "BV1Tv42117jv_q5218_freeze.mp4",
    "X_4EJZ4aors_q9728_freeze.mp4",
    "BV1ZD4y1B7T3_q9604_freeze.mp4",
}

def out_name(video_id, qid):
    return f"{video_id}_q{qid}_freeze{OUTPUT_EXT}"

with open(FILTERED_JSON, encoding="utf-8") as f:
    entries = json.load(f)

matched = []
matched_names = set()

for e in entries:
    name = out_name(e["video_id"], e["qid"])
    if name in BROKEN_BASENAMES:
        matched.append(e)
        matched_names.add(name)

unmatched = BROKEN_BASENAMES - matched_names

print(f"broken 文件总数: {len(BROKEN_BASENAMES)}")
print(f"成功匹配到 entry 的数量: {len(matched)}")
print("未能匹配到任何 entry 的文件名（需要你核对，不要由我猜测原因）：")
for name in sorted(unmatched):
    print(f"  UNMATCHED: {name}")

with open(SUBSET_JSON, "w", encoding="utf-8") as f:
    json.dump(matched, f, ensure_ascii=False, indent=2)

print(f"子集已写入: {SUBSET_JSON}")
#!/usr/bin/env python3
"""
测试抹黑版 build_insufficient.py：验证输出视频的
  (1) 时长是否和源视频一致（无漂移）
  (2) evidence 区间是否真的被抹黑（左上角像素接近全黑）
  (3) 非 evidence 区间是否保留原画面
不修改你的脚本，只调用它（按 entry index），再用 ffprobe/ffmpeg 校验。

用法：
  python test_blackout.py --n 5                # 测前 5 条
  python test_blackout.py --qids 5218 66 79    # 测指定 qid
  python test_blackout.py --n 5 --pick-multi   # 优先挑 evidence 段数多的
"""
import argparse, json, os, subprocess, tempfile, sys

SCRIPT = "scripts/vqa/build_insufficient.py"
JSON   = "cgbench_pipeline/cgbench_filtered.json"
VDIR   = "source_datasets/cg_bench/videos"

def run(c): return subprocess.run(c, capture_output=True, text=True)

def probe_dur(path):
    o = run(["ffprobe","-v","error","-of","json","-show_format","-show_streams",path])
    try: info=json.loads(o.stdout or "{}")
    except: return None,None,None
    fd=None
    try: fd=float(info["format"]["duration"])
    except: pass
    vd=nf=None
    for st in info.get("streams",[]):
        if st.get("codec_type")=="video":
            try: vd=float(st.get("duration"))
            except: pass
            try: nf=int(st.get("nb_frames"))
            except: pass
            break
    return fd,vd,nf

def resolve_input(video_id):
    for f in os.listdir(VDIR):
        if os.path.splitext(f)[0]==video_id:
            return os.path.join(VDIR,f)
    return None

def mean_luma(video, t):
    """抽 video 在 t 秒的一帧，返回左上角 200x200 区域平均亮度（0=全黑）。"""
    with tempfile.NamedTemporaryFile(suffix=".png",delete=False) as tf:
        png=tf.name
    run(["ffmpeg","-y","-i",video,"-ss",str(t),"-frames:v","1",
         "-vf","crop=200:200:0:0","-an",png])
    try:
        from PIL import Image
        import numpy as np
        a=np.asarray(Image.open(png).convert("L"))
        return float(a.mean())
    except Exception:
        # 没有 PIL，用 ffmpeg signalstats 读平均亮度
        o=run(["ffmpeg","-i",png,"-vf","signalstats,metadata=print:key=lavfi.signalstats.YAVG",
               "-f","null","-"])
        for line in o.stderr.splitlines():
            if "YAVG" in line:
                try: return float(line.split("=")[-1])
                except: pass
        return -1.0
    finally:
        try: os.remove(png)
        except: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=5)
    ap.add_argument("--qids",type=int,nargs="+",default=None)
    ap.add_argument("--pick-multi",action="store_true",
                    help="优先挑 evidence 段数多的条目")
    ap.add_argument("--tol",type=float,default=0.5,help="时长容差(秒)")
    ap.add_argument("--out-dir",default="/tmp/blackout_test")
    a=ap.parse_args()

    data=json.load(open(JSON))
    data=[e for e in data if e.get("evidence_intervals")]

    # 选样本
    if a.qids:
        qset=set(a.qids)
        sample=[(i,e) for i,e in enumerate(json.load(open(JSON))) if e.get("qid") in qset]
    else:
        idxd=list(enumerate(json.load(open(JSON))))
        idxd=[(i,e) for i,e in idxd if e.get("evidence_intervals")]
        if a.pick_multi:
            idxd.sort(key=lambda x:len(x[1]["evidence_intervals"]),reverse=True)
        sample=idxd[:a.n]

    os.makedirs(a.out_dir,exist_ok=True)
    print(f"{'qid':>7} {'segs':>4} {'src_dur':>9} {'out_dur':>9} {'diff':>8} "
          f"{'frames':>7} {'ev_luma':>7} {'nonev_luma':>10} {'verdict':>8}")
    print("-"*90)

    all_ok=True
    for idx,e in sample:
        vid=e["video_id"]; qid=e["qid"]
        ivs=e["evidence_intervals"]
        inp=resolve_input(vid)
        if not inp:
            print(f"{qid:>7}  MISSING video {vid}"); all_ok=False; continue
        src_fd,src_vd,_=probe_dur(inp)

        out=os.path.join(a.out_dir,f"{vid}_q{qid}_insufficient.mp4")
        if os.path.exists(out): os.remove(out)
        # 调你的脚本：输出到测试目录，处理这一条
        r=run(["python",SCRIPT,"--index",str(idx),
               "--output-dir",a.out_dir,
               "--x264-preset","veryfast","--crf","18",
               "--report",os.path.join(a.out_dir,"r.jsonl"),
               "--overwrite"])
        if not os.path.exists(out):
            print(f"{qid:>7}  BUILD FAILED: {r.stderr[-200:] if r.stderr else r.stdout[-200:]}")
            all_ok=False; continue

        out_fd,out_vd,out_nf=probe_dur(out)
        diff=(out_fd or 0)-(src_fd or 0)

        # 抽 evidence 中点 与 一个非 evidence 点 的亮度
        es,ee=float(ivs[0]["start"]),float(ivs[0]["end"])
        ev_t=(es+ee)/2
        # 找一个不在任何 evidence 区间内的时刻
        nonev_t=None
        for cand in [5, (src_fd or 60)/2, (src_fd or 60)*0.8]:
            inside=any(float(iv["start"])<=cand<=float(iv["end"]) for iv in ivs)
            if not inside: nonev_t=cand; break
        ev_luma=mean_luma(out,ev_t)
        nonev_luma=mean_luma(out,nonev_t) if nonev_t else -1

        dur_ok = abs(diff)<=a.tol
        black_ok = ev_luma>=0 and ev_luma<10        # evidence 抹黑后应接近全黑
        nonev_ok = nonev_luma<0 or nonev_luma>5     # 非 evidence 不应全黑
        verdict = "OK" if (dur_ok and black_ok and nonev_ok) else "FAIL"
        if verdict=="FAIL": all_ok=False
        print(f"{qid:>7} {len(ivs):>4} {src_fd:>9.3f} {out_fd:>9.3f} {diff:>+8.3f} "
              f"{str(out_nf):>7} {ev_luma:>7.1f} {nonev_luma:>10.1f} {verdict:>8}")

    print("-"*90)
    print("全部通过 ✅" if all_ok else "有问题，看上面的 FAIL 行 ❌")
    print(f"\n生成的测试视频在 {a.out_dir}，可手动播放抽查 evidence 是否抹黑、画面是否对。")
    print("时长判定：|diff| <= tol 即无漂移。evidence 亮度应接近 0(全黑)，非 evidence 应 >5。")

if __name__=="__main__":
    main()
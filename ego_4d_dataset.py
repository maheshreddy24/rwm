import json, os, random
from collections import defaultdict

TARGET_GB    = 200
GB_PER_HOUR  = 0.6      # video_540ss estimate — recalibrate after first run
MIN_SEC, MAX_SEC = 180, 1800

meta = json.load(open(os.path.expanduser("~/ego4d/data/ego4d.json")))
vids = [v for v in meta["videos"] if MIN_SEC < v.get("duration_sec", 0) < MAX_SEC]

by_scen = defaultdict(list)
for v in vids:
    for s in (v.get("scenarios") or ["unknown"]):
        by_scen[s].append(v)

rng = random.Random(0)
pools = {s: rng.sample(l, len(l)) for s, l in by_scen.items()}

target_sec = (TARGET_GB / GB_PER_HOUR) * 3600
seen, total, per_scen = set(), 0.0, defaultdict(int)

while total < target_sec:
    progressed = False
    for s in sorted(pools):                      # one video per scenario per round
        while pools[s]:
            v = pools[s].pop()
            if v["video_uid"] in seen:
                continue
            seen.add(v["video_uid"])
            total += v["duration_sec"]
            per_scen[s] += 1
            progressed = True
            break
        if total >= target_sec:
            break
    if not progressed:                           # pools exhausted
        break

print(f"{len(seen)} videos | {total/3600:.1f} h | ~{total/3600*GB_PER_HOUR:.0f} GB est")
print(f"{len(per_scen)} scenarios, {min(per_scen.values())}-{max(per_scen.values())} videos each\n")
for s, n in sorted(per_scen.items(), key=lambda x: -x[1])[:15]:
    print(f"{n:4d}  {s}")

with open(os.path.expanduser("~/ego4d/uids.txt"), "w") as f:
    f.write("\n".join(sorted(seen)))
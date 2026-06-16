"""Render the tiered-cache long-run monitor into README figures.

Reads tests/tiered_longrun_data.json (produced by tests/bench_tiered_longrun.py)
and writes into assets/:

  tiered_mem.png       stacked area: per-tier cache memory (GPU/CPU/NVMe) over
                       generations, each tier pinned at its bound -> bounded forever.
  tiered_cost.png      per-residual round-trip (put+get) latency per tier (log y):
                       the monotonic cost of tier depth.
  tiered_longrun.gif   animation of the stacked memory filling GPU -> CPU -> NVMe
                       generation by generation, with the per-generation move cost.

Run:  python tests/plot_tiered_longrun.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "tiered_longrun_data.json")
ASSETS = os.path.join(ROOT, "assets")

# tier colours: GPU (green, hot), CPU (amber, warm), NVMe (blue, cold)
C_GPU, C_CPU, C_NVME = "#2e8b57", "#e1a33a", "#3a7bd5"
BG = "#0d1117"; FG = "#c9d1d9"; GRID = "#30363d"


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)


def plot_mem(d):
    t = d["trace"]; cfg = d["config"]; eng = d["engage"]
    g = [r["gen"] for r in t]
    gpu = [r["gpu"] for r in t]; cpu = [r["cpu"] for r in t]; nvme = [r["nvme"] for r in t]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    fig.patch.set_facecolor(BG); style(ax)
    ax.stackplot(g, gpu, cpu, nvme, labels=["GPU", "CPU (RAM)", "NVMe"],
                 colors=[C_GPU, C_CPU, C_NVME], alpha=0.92)
    tot = cfg["gpu_bound_mb"] + cfg["cpu_bound_mb"] + cfg["nvme_bound_mb"]
    ax.axhline(tot, color=FG, lw=1.0, ls="--", alpha=0.7)
    ax.text(g[-1], tot, f"  total cap {tot/1024:.1f} GB", color=FG, va="center", ha="right", fontsize=8)

    for gen, txt in ((eng["cpu"], "CPU\nengages"), (eng["nvme"], "NVMe\nengages"),
                     (eng["evict"], "evict →\nbounded\nforever")):
        if gen:
            ax.axvline(gen, color=FG, lw=0.8, ls=":", alpha=0.5)
            ax.text(gen, tot * 0.02, txt, color=FG, fontsize=7.5, ha="center", va="bottom", alpha=0.85)

    ax.set_xlim(g[0], g[-1]); ax.set_ylim(0, tot * 1.08)
    ax.set_xlabel("generation (continual, fresh gen_id each)"); ax.set_ylabel("cache resident (MB)")
    ax.set_title(f"Tiered cache memory over a long continual run "
                 f"({cfg['per_gen_mb']:.0f} MB/gen, GPU≤{cfg['gpu_bound_mb']:.0f} → "
                 f"CPU≤{cfg['cpu_bound_mb']:.0f} → NVMe≤{cfg['nvme_bound_mb']:.0f} MB)", fontsize=10)
    leg = ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    out = os.path.join(ASSETS, "tiered_mem.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("wrote", out)


def plot_cost(d):
    tc = d["tier_cost"]
    names = [t["tier"] for t in tc]
    put = [t["put_ms"] for t in tc]; get = [t["get_ms"] for t in tc]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=130)
    fig.patch.set_facecolor(BG); style(ax)
    w = 0.38
    ax.bar([i - w / 2 for i in x], put, w, label="put (offload)", color=C_CPU)
    ax.bar([i + w / 2 for i in x], get, w, label="get (fetch)", color=C_NVME)
    ax.set_yscale("log")
    rtmax = max(t["roundtrip_ms"] for t in tc)
    ax.set_ylim(top=rtmax * 6)                       # headroom so labels clear the title
    for i, t in enumerate(tc):
        ax.text(i, t["roundtrip_ms"] * 1.25, f"{t['roundtrip_ms']:.2f} ms\nround trip",
                color=FG, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{n}\n({c})" for n, c in
                  zip(names, ["reference", "H2D/D2H copy", "file + copy"])])
    ax.set_ylabel("latency per residual (ms, log)")
    ax.set_title("Cost of tier depth: one cached residual's round trip", fontsize=10, pad=12)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    out = os.path.join(ASSETS, "tiered_cost.png")
    fig.savefig(out, facecolor=BG); plt.close(fig)
    print("wrote", out)


def animate(d):
    t = d["trace"]; cfg = d["config"]
    g = [r["gen"] for r in t]
    gpu = [r["gpu"] for r in t]; cpu = [r["cpu"] for r in t]; nvme = [r["nvme"] for r in t]
    tot = cfg["gpu_bound_mb"] + cfg["cpu_bound_mb"] + cfg["nvme_bound_mb"]
    mvmax = max(r["move_ms"] for r in t)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=120,
                                  gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(BG); style(ax); style(axb)

    def draw(i):
        ax.clear(); axb.clear(); style(ax); style(axb)
        gg, cc, nn = g[:i + 1], None, None
        ax.stackplot(gg, gpu[:i + 1], cpu[:i + 1], nvme[:i + 1],
                     labels=["GPU", "CPU (RAM)", "NVMe"], colors=[C_GPU, C_CPU, C_NVME], alpha=0.92)
        ax.axhline(tot, color=FG, lw=1.0, ls="--", alpha=0.6)
        ax.set_xlim(g[0], g[-1]); ax.set_ylim(0, tot * 1.08)
        ax.set_xlabel("generation"); ax.set_ylabel("cache resident (MB)")
        ax.set_title("Tiered cache fills GPU → CPU → NVMe, stays bounded", fontsize=10)
        ax.legend(loc="upper left", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)

        r = t[i]
        axb.bar([0, 1, 2], [r["gpu"], r["cpu"], r["nvme"]], color=[C_GPU, C_CPU, C_NVME])
        axb.set_xticks([0, 1, 2]); axb.set_xticklabels(["GPU", "CPU", "NVMe"])
        axb.set_ylim(0, max(cfg["nvme_bound_mb"], cfg["cpu_bound_mb"]) * 1.15)
        axb.set_title(f"gen {r['gen']}  ·  move {r['move_ms']:.0f} ms", fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=len(t), interval=320, blit=False)
    out = os.path.join(ASSETS, "tiered_longrun.gif")
    anim.save(out, writer=PillowWriter(fps=3))
    plt.close(fig)
    print("wrote", out)


def main():
    with open(DATA) as f:
        d = json.load(f)
    os.makedirs(ASSETS, exist_ok=True)
    plot_mem(d)
    plot_cost(d)
    animate(d)


if __name__ == "__main__":
    main()

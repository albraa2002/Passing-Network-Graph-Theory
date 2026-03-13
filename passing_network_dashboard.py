# ============================================================
#  AL AHLY SC — PASSING NETWORK & TACTICAL HUB ANALYSIS
#  Lead Sports Data Scientist | Google Colab Single-Cell
# ============================================================

# ── 0. INSTALL & IMPORTS ─────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "networkx", "--quiet"], check=True)

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from google.colab import files

# ── 1. SIMULATE PASSING DATA ─────────────────────────────────
np.random.seed(42)

players = [
    "El Shenawy",   # GK
    "Hany",         # RB
    "Abdelmonem",   # CB-R
    "Rabia",        # CB-L
    "Maaloul",      # LB
    "Dieng",        # DM  ← hub
    "Marwan",       # CM  ← hub
    "Ashour",       # CM  ← hub
    "Percy Tau",    # RW
    "El Shahat",    # AM / No.10
    "Kahraba",      # LW / striker
]

# Tactical (X, Y) coords — top-down pitch (0-100 wide, 0-68 tall)
# Origin = bottom-left, attack = right
coords = {
    "El Shenawy":  (5,  34),
    "Hany":        (25, 62),
    "Abdelmonem":  (25, 44),
    "Rabia":       (25, 24),
    "Maaloul":     (25,  8),
    "Dieng":       (45, 34),   # DM — deep hub
    "Marwan":      (58, 50),   # CM — hub
    "Ashour":      (58, 18),   # CM — hub
    "Percy Tau":   (72, 60),
    "El Shahat":   (72, 34),
    "Kahraba":     (72, 10),
}

# Hub index: higher = more passes routed through
hub_weight = {
    "El Shenawy": 0.3,
    "Hany": 0.6,
    "Abdelmonem": 0.6,
    "Rabia": 0.6,
    "Maaloul": 0.55,
    "Dieng": 1.0,     # primary hub
    "Marwan": 0.95,   # primary hub
    "Ashour": 0.90,   # primary hub
    "Percy Tau": 0.5,
    "El Shahat": 0.65,
    "Kahraba": 0.45,
}

rows = []
for passer in players:
    for receiver in players:
        if passer == receiver:
            continue
        w = hub_weight[passer] * hub_weight[receiver]
        base = int(w * 28)
        count = max(0, int(np.random.normal(base, base * 0.25)))
        if count > 0:
            rows.append({"Passer": passer, "Receiver": receiver, "Pass_Count": count})

df = pd.DataFrame(rows)
print(f"✅  Pass events generated: {len(df)} edges")
print(df.sort_values("Pass_Count", ascending=False).head(10).to_string(index=False))

# ── 2. GRAPH THEORY ──────────────────────────────────────────
G = nx.DiGraph()
G.add_nodes_from(players)
for _, row in df.iterrows():
    G.add_edge(row["Passer"], row["Receiver"], weight=row["Pass_Count"])

degree_cent     = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)

# KPI derivations
most_influential = max(betweenness_cent, key=betweenness_cent.get)
strongest_pair   = df.sort_values("Pass_Count", ascending=False).iloc[0]
total_passes     = df["Pass_Count"].sum()
avg_passes       = df["Pass_Count"].mean()

print(f"\n🧠  Most Influential (Betweenness): {most_influential}  ({betweenness_cent[most_influential]:.3f})")
print(f"🔗  Strongest Link: {strongest_pair['Passer']} → {strongest_pair['Receiver']}  ({strongest_pair['Pass_Count']} passes)")

# ── 3. VISUAL HELPERS ─────────────────────────────────────────
# Colours
COL_BG          = "#0a0e1a"
COL_CARD        = "#111827"
COL_PITCH_DARK  = "#0d1a12"
COL_PITCH_LINE  = "rgba(255,255,255,0.18)"
COL_EDGE_BASE   = "rgba(250,204,21,"       # gold — opacity appended later
COL_NODE_GRAD   = ["#16a34a", "#facc15", "#ef4444"]   # green→gold→red by centrality
COL_HUB_RING    = "#facc15"
COL_TEXT        = "#f1f5f9"
COL_ACCENT      = "#facc15"

PITCH_W, PITCH_H = 100, 68

# Node sizes — betweenness → radius
bc_vals = np.array([betweenness_cent[p] for p in players])
bc_norm = (bc_vals - bc_vals.min()) / (bc_vals.max() - bc_vals.min() + 1e-9)
node_sizes = 18 + bc_norm * 36        # 18–54 px marker size

# Edge widths / opacity — pass_count → thickness
max_passes = df["Pass_Count"].max()
def edge_style(count):
    t = count / max_passes
    width   = 1 + t * 7
    opacity = 0.12 + t * 0.75
    return width, opacity

# ── 4. BUILD FIGURE ──────────────────────────────────────────
fig = go.Figure()

# 4-A  PITCH BACKGROUND ─────────────────────────────────────
# Full pitch rectangle
fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_W, y1=PITCH_H,
              fillcolor=COL_PITCH_DARK, line=dict(color=COL_PITCH_LINE, width=1.5))

# Half-way line
fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=PITCH_H,
              line=dict(color=COL_PITCH_LINE, width=1.5))

# Centre circle
theta = np.linspace(0, 2*np.pi, 80)
cx, cy, r = 50, 34, 9.15
fig.add_shape(type="circle", x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
              line=dict(color=COL_PITCH_LINE, width=1.5))

# Penalty areas
for side in [(0, 16.5), (83.5, 100)]:
    x0, x1 = side
    fig.add_shape(type="rect", x0=x0, y0=13.84, x1=x1, y1=54.16,
                  line=dict(color=COL_PITCH_LINE, width=1.5))
    # 6-yard box
    if x0 == 0:
        fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16,
                      line=dict(color=COL_PITCH_LINE, width=1.2))
    else:
        fig.add_shape(type="rect", x0=94.5, y0=24.84, x1=100, y1=43.16,
                      line=dict(color=COL_PITCH_LINE, width=1.2))

# Goals
for gx in [(-1.2, 0), (100, 101.2)]:
    fig.add_shape(type="rect", x0=gx[0], y0=27.68, x1=gx[1], y1=40.32,
                  line=dict(color=COL_PITCH_LINE, width=1.8))

# Penalty spots
for px in [11, 89]:
    fig.add_trace(go.Scatter(x=[px], y=[34], mode="markers",
                             marker=dict(size=4, color=COL_PITCH_LINE),
                             showlegend=False, hoverinfo="skip"))

# 4-B  EDGES (pass lanes) ─────────────────────────────────────
for _, row in df.iterrows():
    x0, y0 = coords[row["Passer"]]
    x1, y1 = coords[row["Receiver"]]
    w, op   = edge_style(row["Pass_Count"])
    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(width=w, color=f"rgba(250,204,21,{op:.2f})"),
        hoverinfo="skip",
        showlegend=False,
    ))

# Arrow heads via annotations
for _, row in df.iterrows():
    if row["Pass_Count"] < df["Pass_Count"].quantile(0.75):
        continue   # only show arrows on strongest 25% of links
    x0, y0 = coords[row["Passer"]]
    x1, y1 = coords[row["Receiver"]]
    _, op = edge_style(row["Pass_Count"])
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.5, arrowcolor=f"rgba(250,204,21,{min(op+0.2,1.0):.2f})",
    )

# 4-C  NODES (players) ────────────────────────────────────────
# Colour nodes by degree centrality
dc_vals = np.array([degree_cent[p] for p in players])
dc_norm = (dc_vals - dc_vals.min()) / (dc_vals.max() - dc_vals.min() + 1e-9)

# Map 0→1 to green→gold→red
def cent_color(v):
    if v < 0.5:
        t = v * 2
        r = int(22  + t*(250-22))
        g = int(163 + t*(204-163))
        b = int(74  + t*(21-74))
    else:
        t = (v - 0.5) * 2
        r = int(250 + t*(239-250))
        g = int(204 + t*(68-204))
        b = int(21  + t*(68-21))
    return f"rgb({r},{g},{b})"

node_colors = [cent_color(v) for v in dc_norm]
hover_texts = []
for p in players:
    bc = betweenness_cent[p]
    dc = degree_cent[p]
    out_pass = df[df["Passer"]==p]["Pass_Count"].sum()
    in_pass  = df[df["Receiver"]==p]["Pass_Count"].sum()
    hover_texts.append(
        f"<b>{p}</b><br>"
        f"Betweenness Centrality: {bc:.3f}<br>"
        f"Degree Centrality: {dc:.3f}<br>"
        f"Passes Out: {out_pass}<br>"
        f"Passes In: {in_pass}"
    )

xs = [coords[p][0] for p in players]
ys = [coords[p][1] for p in players]

# Outer glow ring for hubs (top-3 betweenness)
hub_threshold = sorted(betweenness_cent.values(), reverse=True)[2]
for i, p in enumerate(players):
    if betweenness_cent[p] >= hub_threshold:
        fig.add_trace(go.Scatter(
            x=[coords[p][0]], y=[coords[p][1]], mode="markers",
            marker=dict(size=node_sizes[i]+14, color="rgba(250,204,21,0.15)",
                        line=dict(color="rgba(250,204,21,0.55)", width=2)),
            showlegend=False, hoverinfo="skip"
        ))

fig.add_trace(go.Scatter(
    x=xs, y=ys,
    mode="markers+text",
    marker=dict(
        size=node_sizes,
        color=node_colors,
        line=dict(color="rgba(255,255,255,0.55)", width=1.5),
        opacity=0.95,
    ),
    text=["<b>"+p.split()[-1]+"</b>" for p in players],
    textposition="top center",
    textfont=dict(size=11, color=COL_TEXT, family="IBM Plex Mono, monospace"),
    hovertext=hover_texts,
    hoverinfo="text",
    hoverlabel=dict(bgcolor=COL_CARD, bordercolor=COL_ACCENT,
                    font=dict(color=COL_TEXT, size=12, family="IBM Plex Mono")),
    showlegend=False,
))

# 4-D  KPI PANEL ANNOTATIONS ──────────────────────────────────
sp_passer   = strongest_pair["Passer"].split()[-1]
sp_receiver = strongest_pair["Receiver"].split()[-1]
sp_count    = int(strongest_pair["Pass_Count"])

kpi_annotations = [
    # Title banner
    dict(x=50, y=74, xref="x", yref="y",
         text="<b>AL AHLY SC — PASSING NETWORK & TACTICAL HUB ANALYSIS</b>",
         font=dict(size=16, color=COL_ACCENT, family="IBM Plex Mono, monospace"),
         showarrow=False, align="center"),
    dict(x=50, y=71.5, xref="x", yref="y",
         text="Node size = Betweenness Centrality  |  Edge width = Pass Count  |  Gold ring = Top-3 Hubs",
         font=dict(size=9.5, color="rgba(241,245,249,0.6)", family="IBM Plex Mono"),
         showarrow=False, align="center"),

    # KPI 1: Most Influential
    dict(x=14, y=-5.5, xref="x", yref="y",
         text=f"<b>🧠 TACTICAL HEARTBEAT</b><br><span style='color:{COL_ACCENT};font-size:13px'>{most_influential}</span><br>Betweenness: {betweenness_cent[most_influential]:.3f}",
         font=dict(size=10, color=COL_TEXT, family="IBM Plex Mono"),
         showarrow=False, align="center",
         bgcolor="rgba(17,24,39,0.9)", bordercolor=COL_ACCENT, borderwidth=1, borderpad=8),

    # KPI 2: Strongest Link
    dict(x=50, y=-5.5, xref="x", yref="y",
         text=f"<b>🔗 STRONGEST LINK</b><br><span style='color:{COL_ACCENT};font-size:13px'>{sp_passer} → {sp_receiver}</span><br>{sp_count} successful passes",
         font=dict(size=10, color=COL_TEXT, family="IBM Plex Mono"),
         showarrow=False, align="center",
         bgcolor="rgba(17,24,39,0.9)", bordercolor=COL_ACCENT, borderwidth=1, borderpad=8),

    # KPI 3: Total passes
    dict(x=86, y=-5.5, xref="x", yref="y",
         text=f"<b>📊 MATCH VOLUME</b><br><span style='color:{COL_ACCENT};font-size:13px'>{total_passes:,}</span><br>total successful passes",
         font=dict(size=10, color=COL_TEXT, family="IBM Plex Mono"),
         showarrow=False, align="center",
         bgcolor="rgba(17,24,39,0.9)", bordercolor=COL_ACCENT, borderwidth=1, borderpad=8),
]

fig.update_layout(
    annotations=kpi_annotations,
    xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[-12, 77],  showgrid=False, zeroline=False, visible=False,
               scaleanchor="x", scaleratio=0.68),
    plot_bgcolor=COL_BG,
    paper_bgcolor=COL_BG,
    margin=dict(l=20, r=20, t=30, b=20),
    width=1180,
    height=780,
    hoverdistance=18,
)

# ── 5. LEGEND TRACES (dummy) ──────────────────────────────────
for label, col in [("GK / Defenders", "rgb(22,163,74)"),
                   ("Midfielders", "rgb(250,204,21)"),
                   ("Attackers", "rgb(239,68,68)")]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=col),
        name=label,
        showlegend=True,
    ))

fig.update_layout(
    legend=dict(
        x=0.01, y=0.01,
        bgcolor="rgba(17,24,39,0.85)",
        bordercolor=COL_ACCENT, borderwidth=1,
        font=dict(color=COL_TEXT, size=10, family="IBM Plex Mono"),
        orientation="v",
    )
)

# ── 6. ASSEMBLE HTML DASHBOARD ───────────────────────────────
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Al Ahly SC — Passing Network Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:      #0a0e1a;
    --card:    #111827;
    --border:  rgba(250,204,21,0.25);
    --accent:  #facc15;
    --text:    #f1f5f9;
    --muted:   #94a3b8;
    --green:   #16a34a;
    --red:     #ef4444;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    min-height: 100vh;
    padding: 0;
  }}

  /* ── TOP BAR ── */
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 32px;
    border-bottom: 1px solid var(--border);
    background: rgba(17,24,39,0.95);
    position: sticky; top: 0; z-index: 100;
  }}
  .logo-block {{ display: flex; align-items: center; gap: 14px; }}
  .logo-icon {{
    width: 42px; height: 42px; border-radius: 50%;
    background: conic-gradient(var(--accent) 0deg 90deg, var(--red) 90deg 180deg,
                               var(--accent) 180deg 270deg, var(--green) 270deg 360deg);
    display: grid; place-items: center;
    font-size: 18px; font-weight: 700; color: var(--bg);
  }}
  .logo-text {{ font-size: 13px; line-height: 1.35; }}
  .logo-text strong {{ font-size: 15px; color: var(--accent); letter-spacing: 0.08em; }}
  .header-tag {{
    font-size: 10px; letter-spacing: 0.15em; color: var(--muted);
    border: 1px solid var(--border); padding: 5px 12px; border-radius: 2px;
    text-transform: uppercase;
  }}
  .live-badge {{
    display: flex; align-items: center; gap: 7px;
    font-size: 11px; color: var(--green);
    border: 1px solid rgba(22,163,74,0.4); padding: 5px 12px; border-radius: 2px;
  }}
  .live-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1.8s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.4; transform:scale(0.7); }}
  }}

  /* ── MAIN ── */
  main {{ padding: 24px 32px 40px; }}

  /* ── KPI STRIP ── */
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    padding: 16px 20px;
    border-radius: 3px;
    position: relative; overflow: hidden;
  }}
  .kpi-card::after {{
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(250,204,21,0.04) 0%, transparent 60%);
    pointer-events: none;
  }}
  .kpi-label {{ font-size: 9px; letter-spacing: 0.18em; color: var(--muted);
                text-transform: uppercase; margin-bottom: 8px; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; color: var(--accent); line-height: 1; }}
  .kpi-sub   {{ font-size: 10px; color: var(--muted); margin-top: 6px; }}
  .kpi-icon  {{ position: absolute; right: 16px; top: 14px;
                font-size: 20px; opacity: 0.25; }}

  /* ── PITCH PANEL ── */
  .pitch-panel {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 24px;
  }}
  .panel-header {{
    padding: 14px 22px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }}
  .panel-title {{
    font-size: 11px; font-weight: 600; color: var(--text);
    letter-spacing: 0.12em; text-transform: uppercase;
  }}
  .panel-subtitle {{ font-size: 10px; color: var(--muted); }}
  .plotly-wrap {{ padding: 0; }}

  /* ── CENTRALITY TABLE ── */
  .bottom-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }}
  .data-panel {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}
  .data-panel table {{
    width: 100%; border-collapse: collapse;
    font-size: 11px;
  }}
  .data-panel th {{
    padding: 10px 14px;
    text-align: left;
    font-size: 9px; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.3);
  }}
  .data-panel td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(250,204,21,0.07);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
  }}
  .data-panel tr:last-child td {{ border-bottom: none; }}
  .data-panel tr:hover td {{ background: rgba(250,204,21,0.05); }}
  .bar-cell {{ width: 120px; }}
  .bar-bg {{
    height: 5px; background: rgba(255,255,255,0.08);
    border-radius: 2px; overflow: hidden;
  }}
  .bar-fill {{
    height: 100%; background: var(--accent); border-radius: 2px;
  }}
  .rank-badge {{
    display: inline-block;
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(250,204,21,0.12);
    border: 1px solid rgba(250,204,21,0.35);
    text-align: center; line-height: 20px;
    font-size: 9px; color: var(--accent);
  }}

  /* ── FOOTER ── */
  footer {{
    margin-top: 32px;
    padding: 16px 32px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 10px; color: var(--muted); letter-spacing: 0.1em;
  }}
</style>
</head>
<body>

<header>
  <div class="logo-block">
    <div class="logo-icon">⚽</div>
    <div class="logo-text">
      <strong>AL AHLY SC</strong><br>
      <span style="color:var(--muted);font-size:11px">Passing Network · Tactical Hub Analysis</span>
    </div>
  </div>
  <div class="header-tag">Match Intelligence Platform</div>
  <div style="display:flex;gap:12px;align-items:center;">
    <div class="live-badge"><div class="live-dot"></div>LIVE ANALYSIS</div>
    <div class="header-tag">Graph Theory · NetworkX · Plotly</div>
  </div>
</header>

<main>

  <!-- KPI STRIP -->
  <div class="kpi-strip">
    <div class="kpi-card">
      <div class="kpi-icon">🧠</div>
      <div class="kpi-label">Tactical Heartbeat</div>
      <div class="kpi-value">{most_influential.split()[-1]}</div>
      <div class="kpi-sub">Highest Betweenness Centrality<br>{betweenness_cent[most_influential]:.4f} score</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🔗</div>
      <div class="kpi-label">Strongest Link</div>
      <div class="kpi-value">{sp_passer} → {sp_receiver}</div>
      <div class="kpi-sub">{sp_count} successful passes<br>dominant passing corridor</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">📊</div>
      <div class="kpi-label">Total Passes</div>
      <div class="kpi-value">{total_passes:,}</div>
      <div class="kpi-sub">Across {len(df)} unique passing combinations</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">⚡</div>
      <div class="kpi-label">Network Density</div>
      <div class="kpi-value">{nx.density(G):.3f}</div>
      <div class="kpi-sub">Directed graph · {len(players)} nodes · {G.number_of_edges()} edges</div>
    </div>
  </div>

  <!-- PITCH MAP -->
  <div class="pitch-panel">
    <div class="panel-header">
      <div>
        <div class="panel-title">Tactical Passing Network — 4-2-3-1 Formation</div>
        <div class="panel-subtitle">Node size = Betweenness Centrality  ·  Edge width = Pass volume  ·  Gold ring = Top-3 hubs</div>
      </div>
      <div class="panel-subtitle">↑ Attack direction →</div>
    </div>
    <div class="plotly-wrap">
      {plot_html}
    </div>
  </div>

  <!-- CENTRALITY TABLES -->
  <div class="bottom-grid">

    <!-- Betweenness -->
    <div class="data-panel">
      <div class="panel-header">
        <div class="panel-title">Betweenness Centrality Ranking</div>
        <div class="panel-subtitle">Bridge / link-pin role</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Player</th><th>Score</th><th class="bar-cell">Visual</th>
          </tr>
        </thead>
        <tbody>
          {''.join([
              f'<tr><td><span class="rank-badge">{i+1}</span></td>'
              f'<td><b>{p}</b></td>'
              f'<td style="color:var(--accent)">{betweenness_cent[p]:.4f}</td>'
              f'<td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:{betweenness_cent[p]/max(betweenness_cent.values())*100:.1f}%"></div></div></td></tr>'
              for i, p in enumerate(sorted(betweenness_cent, key=betweenness_cent.get, reverse=True))
          ])}
        </tbody>
      </table>
    </div>

    <!-- Degree -->
    <div class="data-panel">
      <div class="panel-header">
        <div class="panel-title">Degree Centrality Ranking</div>
        <div class="panel-subtitle">Overall involvement</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Player</th><th>Score</th><th class="bar-cell">Visual</th>
          </tr>
        </thead>
        <tbody>
          {''.join([
              f'<tr><td><span class="rank-badge">{i+1}</span></td>'
              f'<td><b>{p}</b></td>'
              f'<td style="color:var(--accent)">{degree_cent[p]:.4f}</td>'
              f'<td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:{degree_cent[p]/max(degree_cent.values())*100:.1f}%"></div></div></td></tr>'
              for i, p in enumerate(sorted(degree_cent, key=degree_cent.get, reverse=True))
          ])}
        </tbody>
      </table>
    </div>

  </div><!-- /bottom-grid -->

</main>

<footer>
  AL AHLY SC PERFORMANCE ANALYTICS DIVISION  ·  PASSING NETWORK ENGINE v2.0  ·  Graph Theory via NetworkX  ·  Visualised with Plotly
</footer>

</body>
</html>"""

# ── 7. SAVE & DOWNLOAD ───────────────────────────────────────
output_path = "Passing_Network_Dashboard.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"\n✅  Dashboard saved → {output_path}")
print(f"📥  Initiating download...")
files.download(output_path)
print("🏆  Done! Al Ahly SC Passing Network Dashboard is ready.")

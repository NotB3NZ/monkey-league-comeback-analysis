"""
Monkey League Season 6 Grand Final Analysis


Primary research question:
  "How did a 4–1 Xuanyi lead turn into a 5–4 Yiheng victory?"

This script produces five primary visualizations, supporting statistics,
and a full analytical narrative. Redesigned for a light-mode editorial aesthetic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import median_abs_deviation
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# DESIGN
# Player colors
YIHENG_COLOR = "#2563EB"      # Blue
XUANYI_COLOR = "#DC2626"      # Red
YIHENG_LIGHT = "#DBEAFE"      # Very light blue for fills/shading
XUANYI_LIGHT = "#FEE2E2"      # Very light red for fills/shading

# Neutral colors
BG_COLOR = "#FAFAFA"          # Very light warm-gray/white
CARD_COLOR = "#FFFFFF"        # Pure white for table backgrounds
GRID_COLOR = "#E5E7EB"        # Light gray for subtle gridlines
TEXT_COLOR = "#1F2937"        # Dark charcoal
MUTED_TEXT = "#6B7280"        # Medium gray
SET_BOUNDARY_COLOR = "#D1D5DB"
PHASE_LINE_COLOR = "#4B5563"  # Dark gray for 4-1 boundary
ROLLING_MEDIAN_COLOR = "#374151" # Neutral dark charcoal
UNUSED_CELL_COLOR = "#F3F4F6" # Light gray for empty waffle cells

# Font
FONT_FAMILY = "sans-serif"
TITLE_SIZE = 16
SUBTITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9
ANNOTATION_SIZE = 8

def apply_style():
    """Apply consistent light editorial theme across all plots."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "font.family": FONT_FAMILY,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.6,
        "legend.facecolor": BG_COLOR,
        "legend.edgecolor": BG_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.facecolor": BG_COLOR,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
    })

apply_style()



# DATA LOADING & DERIVED VARIABLES

def load_and_prepare_data(filepath):
    """Raw CSV and create all derived variables."""
    df = pd.read_csv(filepath)
    df["race_number"] = range(1, len(df) + 1)
    df["abs_margin"] = df["Margin"].abs()
    df["phase"] = df["Set"].apply(lambda s: "Before 4-1" if s <= 5 else "Comeback")
    df["yiheng_penalty"] = (
        df["Yiheng_final"].notna() & df["Yiheng"].notna() &
        (abs(df["Yiheng_final"] - df["Yiheng"] - 2.0) < 0.005)
    )
    df["xuanyi_penalty"] = (
        df["Xuanyi_final"].notna() & df["Xuanyi"].notna() &
        (abs(df["Xuanyi_final"] - df["Xuanyi"] - 2.0) < 0.005)
    )
    
    set_winners = {}
    for s in df["Set"].unique():
        sdf = df[df["Set"] == s]
        yw = (sdf["Winner"] == "Yiheng").sum()
        xw = (sdf["Winner"] == "Xuanyi").sum()
        set_winners[s] = "Yiheng" if yw > xw else "Xuanyi"
    df["set_winner"] = df["Set"].map(set_winners)
    
    y_sets, x_sets = 0, 0
    match_scores = {}
    for s in sorted(df["Set"].unique()):
        if set_winners[s] == "Yiheng":
            y_sets += 1
        else:
            x_sets += 1
        match_scores[s] = f"{y_sets}–{x_sets}"
    df["match_score_after_set"] = df["Set"].map(match_scores)
    
    df["rolling_margin"] = df["Margin"].rolling(window=3, min_periods=2, center=True).median()
    return df


def get_set_boundaries(df):
    boundaries = []
    for s in sorted(df["Set"].unique())[:-1]:
        last_race = df[df["Set"] == s]["race_number"].max()
        boundaries.append(last_race + 0.5)
    return boundaries


def get_phase_boundary(df):
    return df[df["Set"] == 5]["race_number"].max() + 0.5


def draw_set_boundaries(ax, df):
    """Add subtle set boundary lines and S1-S9 labels."""
    boundaries = get_set_boundaries(df)
    phase_b = get_phase_boundary(df)
    
    for b in boundaries:
        if abs(b - phase_b) > 0.1:
            ax.axvline(x=b, color=SET_BOUNDARY_COLOR, linestyle="-", linewidth=0.8, alpha=0.8, zorder=0)
            
    # Phase boundary
    ax.axvline(x=phase_b, color=PHASE_LINE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    
    # Set labels
    for s in sorted(df["Set"].unique()):
        sdf = df[df["Set"] == s]
        mid = sdf["race_number"].median()
        ax.text(mid, 1.02, f"S{s}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=TICK_SIZE, color=MUTED_TEXT)
        
    return phase_b

def annotate_match_events(ax, df, y_col_y="Yiheng_final", y_col_x="Xuanyi_final", is_margin=False, y_cap=None):
    #Annotations for timeouts, DNFs, and +2s.
    # Timeouts
    timeouts = df[df["Timeout_before"] == 1]
    for _, row in timeouts.iterrows():
        timeout_x = row["race_number"] - 0.5
        color = YIHENG_COLOR if row['timeout_by'] == 'Yiheng' else XUANYI_COLOR
        
        if is_margin:
            y1 = df.loc[row.name - 1, "Margin"] if row.name > 0 else 0
            y2 = row["Margin"]
            if pd.isna(y1): y1 = 0
            if pd.isna(y2): y2 = 0
            mid_y = (y1 + y2) / 2
            if y_cap is not None:
                mid_y = min(max(mid_y, -y_cap), y_cap)
                
            offset = 15 if mid_y >= 0 else -15
            ax.annotate("TO", (timeout_x, mid_y),
                        xytext=(0, offset), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
                        fontsize=ANNOTATION_SIZE, color=color, ha="center", va="center")
        else:
            y1_y = df.loc[row.name - 1, y_col_y] if row.name > 0 and pd.notna(df.loc[row.name - 1, y_col_y]) else 0
            y1_x = df.loc[row.name - 1, y_col_x] if row.name > 0 and pd.notna(df.loc[row.name - 1, y_col_x]) else 0
            y1 = max(y1_y, y1_x)
            
            y2_y = row[y_col_y] if pd.notna(row[y_col_y]) else 0
            y2_x = row[y_col_x] if pd.notna(row[y_col_x]) else 0
            y2 = max(y2_y, y2_x)
            
            mid_y = (y1 + y2) / 2
            if y_cap is not None and mid_y > y_cap:
                mid_y = y_cap - 0.5
                
            ax.annotate("TO", (timeout_x, mid_y),
                        xytext=(0, 15), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
                        fontsize=ANNOTATION_SIZE, color=color, ha="center", va="center")

    # DNFs
    dnfs = df[(df["Yiheng_DNF"] == 1) | (df["Xuanyi_DNF"] == 1)]
    for _, row in dnfs.iterrows():
        if is_margin:
            ax.text(row["race_number"], 0, "DNF\n×", fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, ha="center", va="center")
        else:
            y_val = row[y_col_x] if row["Yiheng_DNF"] == 1 else row[y_col_y]
            if y_cap is not None and y_val > y_cap: y_val = y_cap - 1
            ax.text(row["race_number"], y_val + 0.5, "DNF\n×", fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, ha="center")

    # +2 Penalties
    pens = df[df["yiheng_penalty"] | df["xuanyi_penalty"]]
    for _, row in pens.iterrows():
        if is_margin:
            y_val = row["Margin"]
            if pd.notna(y_val):
                if y_cap is not None and abs(y_val) > y_cap:
                    # Don't annotate clipped values with +2 if it clutters the top
                    continue
                ax.text(row["race_number"], y_val + 0.3, "+2", fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, ha="center")
        else:
            if row["yiheng_penalty"] and pd.notna(row[y_col_y]):
                y_val = row[y_col_y]
                if y_cap is not None and y_val > y_cap: continue
                ax.text(row["race_number"], y_val + 0.2, "+2", fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, ha="center")
            if row["xuanyi_penalty"] and pd.notna(row[y_col_x]):
                y_val = row[y_col_x]
                if y_cap is not None and y_val > y_cap: continue
                ax.text(row["race_number"], y_val + 0.2, "+2", fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, ha="center")



# SUMMARY STATISTICS FUNCTIONS

def calc_player_stats(times, label=""):
    valid = times.dropna()
    n = len(valid)
    if n == 0: return {}
    return {
        "n_valid": n, "median": valid.median(), "mean": valid.mean(),
        "std": valid.std(), "iqr": valid.quantile(0.75) - valid.quantile(0.25),
        "mad": median_abs_deviation(valid, nan_policy="omit"),
        "best": valid.min(), "worst": valid.max(),
        "sub4_pct": (valid < 4.0).sum() / n * 100, "over5_pct": (valid >= 5.0).sum() / n * 100,
    }

def calc_head_to_head_stats(df_sub):
    valid_margins = df_sub["Margin"].dropna()
    n = len(valid_margins)
    if n == 0: return {}
    return {
        "n_races_with_margin": n, "median_margin": valid_margins.median(),
        "mean_margin": valid_margins.mean(), "median_abs_margin": valid_margins.abs().median(),
    }


# VISUALIZATION 1: SOLVE TIMES


def plot_solve_times(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    Y_MIN, Y_MAX = 3, 9  # Sensible range for the normal variation
    
    # Grid and boundaries
    ax.yaxis.grid(True)
    phase_b = draw_set_boundaries(ax, df)
    
    # Clip data for visualization
    valid_y = df[df["Yiheng_final"].notna()].copy()
    valid_x = df[df["Xuanyi_final"].notna()].copy()
    
    y_clipped = valid_y["Yiheng_final"].clip(upper=Y_MAX)
    x_clipped = valid_x["Xuanyi_final"].clip(upper=Y_MAX)
    
    # Lines (secondary)
    ax.plot(valid_y["race_number"], y_clipped, color=YIHENG_COLOR, linewidth=1, alpha=0.5, zorder=2)
    ax.plot(valid_x["race_number"], x_clipped, color=XUANYI_COLOR, linewidth=1, alpha=0.5, zorder=2)
    
    # Points (primary)
    ax.scatter(valid_y["race_number"], y_clipped, color=YIHENG_COLOR, s=25, zorder=3, label="Yiheng")
    ax.scatter(valid_x["race_number"], x_clipped, color=XUANYI_COLOR, s=25, zorder=3, label="Xuanyi")
    
    # Annotate outlier
    outliers_y = valid_y[valid_y["Yiheng_final"] > Y_MAX]
    for _, row in outliers_y.iterrows():
        ax.annotate(f"↑ {row['Yiheng_final']:.2f}s", (row["race_number"], Y_MAX),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=ANNOTATION_SIZE, color=YIHENG_COLOR, fontweight="bold")
    
    annotate_match_events(ax, df, y_cap=Y_MAX)
    
    # Phase label
    ax.text(phase_b, ax.get_ylim()[1], "Xuanyi leads 4–1", 
            fontsize=ANNOTATION_SIZE, color=PHASE_LINE_COLOR, ha="center", va="bottom")
    
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_xticks(range(1, len(df) + 1, 2))
    ax.set_ylabel("Solve Time (seconds)", fontsize=LABEL_SIZE)
    ax.set_title("Solve Times Across the Match", fontsize=TITLE_SIZE, fontweight="bold", pad=20, loc="left")
    ax.legend(fontsize=TICK_SIZE, loc="upper right", frameon=False, ncol=2)
    
    plt.savefig(output_path)
    plt.close()

def plot_margin_by_race(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    Y_MIN_MAIN, Y_MAX_MAIN = -5, 5
    ax.axhline(0, color=TEXT_COLOR, linewidth=1, alpha=0.8, zorder=1)
    
    phase_b = draw_set_boundaries(ax, df)
    
    df_plot = df.copy()
    df_plot["Margin_clipped"] = df_plot["Margin"].clip(lower=Y_MIN_MAIN, upper=Y_MAX_MAIN)
    
    valid_mask = df_plot["Margin_clipped"].notna()
    group_id = (~valid_mask).cumsum()
    
    for g_id, group in df_plot[valid_mask].groupby(group_id):
        x_vals = group["race_number"].values
        y_vals = group["Margin_clipped"].values
        
        ax.plot(x_vals, y_vals, color=TEXT_COLOR, linewidth=1, alpha=0.6, zorder=3)
        ax.fill_between(x_vals, 0, y_vals, where=(y_vals >= 0), 
                        facecolor=XUANYI_COLOR, alpha=0.3, interpolate=True, zorder=2)
        ax.fill_between(x_vals, 0, y_vals, where=(y_vals <= 0), 
                        facecolor=YIHENG_COLOR, alpha=0.3, interpolate=True, zorder=2)
    
    # Annotate outlier
    outliers = df_plot[df_plot["Margin"] > Y_MAX_MAIN]
    for _, row in outliers.iterrows():
        ax.annotate(f"↑ +{row['Margin']:.1f}s actual margin", (row["race_number"], Y_MAX_MAIN),
                         xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                         fontsize=ANNOTATION_SIZE, color=TEXT_COLOR, fontweight="bold")
        
    annotate_match_events(ax, df, is_margin=True, y_cap=Y_MAX_MAIN)
    
    # Advantage labels (minimal)
    ax.text(1.01, 0.95, "Xuanyi faster", transform=ax.transAxes, color=XUANYI_COLOR, fontsize=TICK_SIZE, va="top")
    ax.text(1.01, 0.05, "Yiheng faster", transform=ax.transAxes, color=YIHENG_COLOR, fontsize=TICK_SIZE, va="bottom")
    
    # Phase label
    ax.text(phase_b, Y_MAX_MAIN, "Xuanyi leads 4–1", 
                 fontsize=ANNOTATION_SIZE, color=PHASE_LINE_COLOR, ha="center", va="bottom")
    
    ax.set_ylim(Y_MIN_MAIN, Y_MAX_MAIN)
    ax.set_ylabel("Margin (s)\n(Yiheng − Xuanyi)", fontsize=LABEL_SIZE)
    ax.set_title("Head-to-Head Margins",
                      fontsize=TITLE_SIZE, fontweight="bold", pad=17.5, loc="left", x=0.011)
                      
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_xticks(range(1, len(df) + 1, 2))
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_rolling_median_margin(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.axhline(0, color=TEXT_COLOR, linewidth=1, alpha=0.8, zorder=1)
    
    phase_b = draw_set_boundaries(ax, df)
    
    # Determine symmetric Y scale based on rolling median data
    max_roll = df["rolling_margin"].abs().max()
    y_lim_roll = np.ceil(max_roll * 2) / 2 if pd.notna(max_roll) and max_roll > 0 else 1.0
    y_lim_roll = max(1.0, y_lim_roll)
    Y_MIN_ROLL, Y_MAX_ROLL = -y_lim_roll, y_lim_roll
    
    valid_mask = df["rolling_margin"].notna()
    group_id = (~valid_mask).cumsum()
    
    for g_id, group in df[valid_mask].groupby(group_id):
        x_vals = group["race_number"].values
        y_vals = group["rolling_margin"].values
        
        ax.plot(x_vals, y_vals, color=TEXT_COLOR, linewidth=1, alpha=0.5, zorder=3)
        ax.fill_between(x_vals, 0, y_vals, where=(y_vals >= 0), 
                        facecolor=XUANYI_COLOR, alpha=0.2, interpolate=True, zorder=2)
        ax.fill_between(x_vals, 0, y_vals, where=(y_vals <= 0), 
                        facecolor=YIHENG_COLOR, alpha=0.2, interpolate=True, zorder=2)
                        
    ax.text(phase_b, Y_MAX_ROLL, "Xuanyi leads 4–1", 
            fontsize=ANNOTATION_SIZE, color=PHASE_LINE_COLOR, ha="center", va="bottom")
            
    ax.set_ylim(Y_MIN_ROLL, Y_MAX_ROLL)
    ax.set_xlabel("Race Number", fontsize=LABEL_SIZE)
    ax.set_ylabel("Rolling Margin (s)", fontsize=LABEL_SIZE)
    ax.set_title("Rolling Median", 
                 fontsize=TITLE_SIZE, fontweight="bold", pad=17, loc="left", x=-0.008)
    
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_xticks(range(1, len(df) + 1, 2))
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# VISUALIZATION 3: DISTRIBUTION PLOTS

def plot_distributions(df, output_path):
    Y_MIN, Y_MAX = 3, 7 # sensible range for boxplots
    
    records = []
    for _, row in df.iterrows():
        if pd.notna(row["Yiheng_final"]):
            records.append({"Player": "Yiheng", "Phase": row["phase"], "Time": row["Yiheng_final"], "grp": f"Yiheng\n{row['phase']}"})
        if pd.notna(row["Xuanyi_final"]):
            records.append({"Player": "Xuanyi", "Phase": row["phase"], "Time": row["Xuanyi_final"], "grp": f"Xuanyi\n{row['phase']}"})
    
    long_df = pd.DataFrame(records)
    group_order = ["Yiheng\nBefore 4-1", "Yiheng\nComeback", "Xuanyi\nBefore 4-1", "Xuanyi\nComeback"]
    colors = [YIHENG_COLOR, YIHENG_COLOR, XUANYI_COLOR, XUANYI_COLOR]
    alphas = [0.4, 0.8, 0.4, 0.8]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.yaxis.grid(True)
    
    bp_data = [long_df[long_df["grp"] == g]["Time"].values for g in group_order]
    
    # Filter for calculation to not distort boxplot visually if using default, but we want 18.45 included in stats.
    # matplotlib boxplot handles outliers (fliers). We will hide fliers and plot jittered points ourselves.
    
    bp = ax.boxplot(bp_data, positions=range(len(group_order)), widths=0.4, patch_artist=True,
                    medianprops=dict(color=BG_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR, linewidth=1),
                    capprops=dict(color=TEXT_COLOR, linewidth=1),
                    boxprops=dict(linewidth=1, edgecolor=TEXT_COLOR),
                    showfliers=False)
    
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(alphas[i])
        
    # Jittered points
    for i, g in enumerate(group_order):
        data = long_df[long_df["grp"] == g]["Time"].values
        # Only plot points within Y_MAX in normal range
        normal_data = data[data <= Y_MAX]
        jitter = np.random.uniform(-0.15, 0.15, size=len(normal_data))
        ax.scatter(i + jitter, normal_data, color=colors[i], s=20, alpha=0.9, edgecolor="white", linewidth=0.5, zorder=5)
        
        # Annotate outliers
        outliers = data[data > Y_MAX]
        for out in outliers:
            ax.annotate(f"↑ {out:.2f}s", (i, Y_MAX), xytext=(0, 5), textcoords="offset points", 
                        ha="center", va="bottom", fontsize=ANNOTATION_SIZE, color=colors[i], fontweight="bold")
    
    ax.axvline(1.5, color=GRID_COLOR, linestyle="-", linewidth=1)
    
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, fontsize=TICK_SIZE)
    ax.set_ylabel("Solve Time (seconds)", fontsize=LABEL_SIZE)
    ax.set_title("Solve-time distributions", fontsize=TITLE_SIZE, fontweight="bold", pad=20, loc="left")
    
    plt.savefig(output_path)
    plt.close()


# VISUALIZATION 4: RACE MATRIX

def plot_race_matrix(df, output_path):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.axis("off")
    
    COLS, ROWS = 6, 7
    total_cells = COLS * ROWS
    n_races = len(df)
    
    # Calculate cell parameters
    cell_w = 1.0
    cell_h = 1.0
    gap = 0.1
    
    # Store coordinates for set boundaries if needed
    set_boundaries = []
    
    for i in range(total_cells):
        row = i // COLS
        col = i % COLS
        
        x = col * (cell_w + gap)
        y = (ROWS - 1 - row) * (cell_h + gap) # draw top to bottom
        
        if i < n_races:
            race_data = df.iloc[i]
            color = YIHENG_COLOR if race_data["Winner"] == "Yiheng" else XUANYI_COLOR
        else:
            color = UNUSED_CELL_COLOR
            
        rect = plt.Rectangle((x, y), cell_w, cell_h, facecolor=color, edgecolor=BG_COLOR, linewidth=1)
        ax.add_patch(rect)
    
    # Overall plot bounds
    ax.set_xlim(-0.5, COLS * (cell_w + gap) + 0.5)
    ax.set_ylim(-1, ROWS * (cell_h + gap) + 1.5)
    
    y_wins = (df["Winner"] == "Yiheng").sum()
    x_wins = (df["Winner"] == "Xuanyi").sum()
    
    # Legend
    legend_y = -0.5
    rect_y = plt.Rectangle((0, legend_y), 0.4, 0.4, facecolor=YIHENG_COLOR)
    ax.add_patch(rect_y)
    ax.text(0.6, legend_y + 0.2, f"Yiheng ({y_wins})", va="center", fontsize=LABEL_SIZE)
    
    rect_x = plt.Rectangle((2.5, legend_y), 0.4, 0.4, facecolor=XUANYI_COLOR)
    ax.add_patch(rect_x)
    ax.text(3.1, legend_y + 0.2, f"Xuanyi ({x_wins})", va="center", fontsize=LABEL_SIZE)
    
    rect_u = plt.Rectangle((5.0, legend_y), 0.4, 0.4, facecolor=UNUSED_CELL_COLOR)
    ax.add_patch(rect_u)
    ax.text(5.6, legend_y + 0.2, "No race", va="center", fontsize=LABEL_SIZE)
    
    ax.set_title("Race Matrix", fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=15)
    ax.text(0, ROWS*(cell_h+gap) + 0.2, f"37 races across 9 sets", fontsize=SUBTITLE_SIZE, color=MUTED_TEXT, ha="left")
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# VISUALIZATION 5: BEFORE/AFTER TABLE

def plot_analytics_table(df, output_path):
    phases = ["Before 4-1", "Comeback"]
    metrics_labels = [
        "Valid Solves", "Median (s)", "Mean (s)", "Std Dev (s)", "IQR (s)",
        "MAD (s)", "Best (s)", "Worst (s)", "Race Win %", "Sub-4s %",
        "5+ s %", "DNFs", "+2 Penalties"
    ]
    
    player_data = {}
    for player, col_final, dnf_col, pen_col in [("Yiheng", "Yiheng_final", "Yiheng_DNF", "yiheng_penalty"),
                                                ("Xuanyi", "Xuanyi_final", "Xuanyi_DNF", "xuanyi_penalty")]:
        player_data[player] = {}
        for phase in phases:
            sub = df[df["phase"] == phase]
            times = sub[col_final].dropna()
            n, total, wins = len(times), len(sub), (sub["Winner"] == player).sum()
            vals = [
                f"{n}", f"{times.median():.2f}" if n > 0 else "–", f"{times.mean():.2f}" if n > 0 else "–",
                f"{times.std():.2f}" if n > 1 else "–", f"{(times.quantile(0.75)-times.quantile(0.25)):.2f}" if n > 0 else "–",
                f"{median_abs_deviation(times, nan_policy='omit'):.2f}" if n > 0 else "–",
                f"{times.min():.2f}" if n > 0 else "–", f"{times.max():.2f}" if n > 0 else "–",
                f"{wins/total*100:.1f}%" if total > 0 else "–", f"{(times < 4.0).sum()/n*100:.1f}%" if n > 0 else "–",
                f"{(times >= 5.0).sum()/n*100:.1f}%" if n > 0 else "–", f"{(sub[dnf_col] == 1).sum()}", f"{int(sub[pen_col].sum())}"
            ]
            player_data[player][phase] = vals
            
    h2h_labels = ["Median Margin (s)", "Mean Margin (s)", "Median |Margin| (s)"]
    h2h_vals = {}
    for phase in phases:
        margins = df[df["phase"] == phase]["Margin"].dropna()
        h2h_vals[phase] = [f"{margins.median():+.2f}" if len(margins)>0 else "–", 
                           f"{margins.mean():+.2f}" if len(margins)>0 else "–", 
                           f"{margins.abs().median():.2f}" if len(margins)>0 else "–"]
                           
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.axis("off")
    ax.set_title("Comeback Analysis", fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=40)
    
    col_labels = ["Metric", "Before 4-1\n(Sets 1–5)", "Comeback\n(Sets 6–9)", "Change"]
    table_data, cell_colors = [], []
    
    def add_rows(title, labels, data_dict, color):
        table_data.append([title, "", "", ""])
        cell_colors.append([color]*4)
        for i, label in enumerate(labels):
            if isinstance(data_dict, dict) and "Before 4-1" in data_dict:
                b_val, c_val = data_dict["Before 4-1"][i], data_dict["Comeback"][i]
            else:
                b_val, c_val = data_dict[0][i], data_dict[1][i]
                
            try:
                b_num = float(b_val.replace("%", "").replace("+",""))
                c_num = float(c_val.replace("%", "").replace("+",""))
                delta = f"{c_num-b_num:+.1f}pp" if "%" in b_val else f"{c_num-b_num:+.2f}"
            except: delta = "–"
            table_data.append([label, b_val, c_val, delta])
            cell_colors.append([CARD_COLOR]*4)

    add_rows("YIHENG WANG", metrics_labels, player_data["Yiheng"], YIHENG_LIGHT)
    add_rows("XUANYI GENG", metrics_labels, player_data["Xuanyi"], XUANYI_LIGHT)
    add_rows("HEAD-TO-HEAD", h2h_labels, h2h_vals, GRID_COLOR)
    
    table = ax.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if key[0] == 0: cell.set_text_props(fontweight="bold")
        if key[0] > 0 and table_data[key[0]-1][0] in ["YIHENG WANG", "XUANYI GENG", "HEAD-TO-HEAD"]:
            cell.set_text_props(fontweight="bold", color=TEXT_COLOR)
            cell.set_edgecolor(CARD_COLOR) # cleaner section headers
            
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# MAIN

def main():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "grand_final.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    df = load_and_prepare_data(raw_path)
    
    plot_race_matrix(df, os.path.join(figures_dir, "race_matrix.png"))
    plot_solve_times(df, os.path.join(figures_dir, "solve_times.png"))
    plot_margin_by_race(df, os.path.join(figures_dir, "margin_by_race.png"))
    plot_rolling_median_margin(df, os.path.join(figures_dir, "rolling_median_margin.png"))
    plot_analytics_table(df, os.path.join(figures_dir, "comeback_analysis.png"))
    
    import reports
    reports.run_reports(df, output_dir)

if __name__ == "__main__":
    main()

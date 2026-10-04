import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import median_abs_deviation, gaussian_kde, norm

# Imports from analysis for consistent design
from analysis import (
    YIHENG_COLOR, XUANYI_COLOR, BG_COLOR, TEXT_COLOR, GRID_COLOR, MUTED_TEXT,
    FONT_FAMILY, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, ANNOTATION_SIZE
)

def calculate_player_stats(df, player, phase=None):
    if phase:
        sub = df[df["phase"] == phase]
    else:
        sub = df
        
    col_final = f"{player}_final"
    dnf_col = f"{player}_DNF"
    pen_col = f"{player.lower()}_penalty"
    
    valid_solves = sub[col_final].dropna()
    n_valid = len(valid_solves)
    total_races = len(sub)
    wins = (sub["Winner"] == player).sum()
    
    # Thresholds
    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
    
    stats = {
        "Total Races": total_races,
        "Valid Solves": n_valid,
        "DNFs": (sub[dnf_col] == 1).sum(),
        "+2 Penalties": int(sub[pen_col].sum()),
        "Timeouts Taken": (sub["timeout_by"] == player).sum(),
        "Race Wins": wins,
        "Race Win %": wins / total_races * 100 if total_races > 0 else 0,
    }
    
    if n_valid > 0:
        stats.update({
            "Mean (s)": valid_solves.mean(),
            "Median (s)": valid_solves.median(),
            "Std Dev (s)": valid_solves.std() if n_valid > 1 else np.nan,
            "IQR (s)": valid_solves.quantile(0.75) - valid_solves.quantile(0.25),
            "MAD (s)": median_abs_deviation(valid_solves, nan_policy='omit'),
            "Best (s)": valid_solves.min(),
            "Worst (s)": valid_solves.max(),
            "Best Race": sub.loc[valid_solves.idxmin(), "race_number"],
            "Worst Race": sub.loc[valid_solves.idxmax(), "race_number"],
        })
        
        for t in thresholds:
            stats[f"Sub-{t}s %"] = (valid_solves < t).sum() / n_valid * 100
            stats[f"Sub-{t}s Count"] = (valid_solves < t).sum()
        stats["5.0+s %"] = (valid_solves >= 5.0).sum() / n_valid * 100
        stats["5.0+s Count"] = (valid_solves >= 5.0).sum()
        stats["6.0+s %"] = (valid_solves >= 6.0).sum() / n_valid * 100
        stats["6.0+s Count"] = (valid_solves >= 6.0).sum()
        stats["7.0+s %"] = (valid_solves >= 7.0).sum() / n_valid * 100
        stats["7.0+s Count"] = (valid_solves >= 7.0).sum()
    else:
        for key in ["Mean (s)", "Median (s)", "Std Dev (s)", "IQR (s)", "MAD (s)", "Best (s)", "Worst (s)"]:
            stats[key] = np.nan
        stats["Best Race"] = None
        stats["Worst Race"] = None
        for t in thresholds:
            stats[f"Sub-{t}s %"] = 0
            stats[f"Sub-{t}s Count"] = 0
        stats["5.0+s %"] = 0
        stats["5.0+s Count"] = 0
        stats["6.0+s %"] = 0
        stats["6.0+s Count"] = 0
        stats["7.0+s %"] = 0
        stats["7.0+s Count"] = 0
        
    return stats

def calculate_head_to_head_stats(df, phase=None):
    if phase:
        sub = df[df["phase"] == phase]
    else:
        sub = df
        
    margins = sub["Margin"].dropna()
    abs_margins = sub["abs_margin"].dropna()
    
    stats = {
        "Yiheng Wins": (sub["Winner"] == "Yiheng").sum(),
        "Xuanyi Wins": (sub["Winner"] == "Xuanyi").sum(),
    }
    
    if len(margins) > 0:
        stats.update({
            "Median Margin (s)": margins.median(),
            "Mean Margin (s)": margins.mean(),
            "Median |Margin| (s)": abs_margins.median(),
            "Mean |Margin| (s)": abs_margins.mean(),
            "Closest Race": sub.loc[abs_margins.idxmin(), "race_number"],
            "Closest |Margin|": abs_margins.min(),
            "Largest Race": sub.loc[abs_margins.idxmax(), "race_number"],
            "Largest |Margin|": abs_margins.max(),
        })
    else:
        for key in ["Median Margin (s)", "Mean Margin (s)", "Median |Margin| (s)", "Mean |Margin| (s)"]:
            stats[key] = np.nan
        stats["Closest Race"] = None
        stats["Closest |Margin|"] = np.nan
        stats["Largest Race"] = None
        stats["Largest |Margin|"] = np.nan
        
    return stats

def format_val(val, fmt="{:.2f}", is_pct=False, is_margin=False):
    if pd.isna(val) or val is None:
        return "–"
    res = fmt.format(val)
    if is_margin and val > 0 and val != 0:
        res = "+" + res
    if is_pct:
        res += "%"
    return res

def format_change(before, after, is_pct=False):
    if pd.isna(before) or pd.isna(after) or before is None or after is None:
        return "–"
    diff = after - before
    if is_pct:
        return f"{diff:+.1f}pp"
    return f"{diff:+.2f}s"

def format_change_int(before, after):
    if pd.isna(before) or pd.isna(after) or before is None or after is None:
        return "–"
    diff = after - before
    return f"{diff:+d}"

def calculate_streaks(df, player):
    wins = (df["Winner"] == player).astype(int)
    if wins.sum() == 0:
        return 0, []
    block = (wins != wins.shift()).cumsum()
    streaks = wins.groupby(block).sum()
    max_streak = int(streaks.max())
    max_blocks = streaks[streaks == max_streak].index
    ranges = []
    for b in max_blocks:
        indices = df[block == b].index
        ranges.append((df.loc[indices[0], "race_number"], df.loc[indices[-1], "race_number"]))
    return max_streak, ranges

def plot_solve_distributions(df, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1, top=0.82)
    
    players = ["Yiheng", "Xuanyi"]
    colors = [YIHENG_COLOR, XUANYI_COLOR]
    phases = [None, "Before 4-1", "Comeback"]
    phase_titles = ["OVERALL", "BEFORE 4–1\nSets 1–5", "COMEBACK\nSets 6–9"]
    
    x_min, x_max = 3.0, 7.5
    bins = np.arange(x_min, x_max + 0.1, 0.25)
    xs = np.linspace(x_min, x_max, 200)
    
    for row, (player, color) in enumerate(zip(players, colors)):
        col_final = f"{player}_final"
        for col, phase in enumerate(phases):
            ax = axes[row, col]
            ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.6)
            
            if phase is None:
                valid = df[col_final].dropna()
            else:
                valid = df[df["phase"] == phase][col_final].dropna()
                
            n = len(valid)
            if n > 0:
                ax.hist(valid, bins=bins, density=True, color=color, edgecolor='none', alpha=0.3)
                
                try:
                    kde = gaussian_kde(valid)
                    ys_kde = kde(xs)
                    ax.plot(xs, ys_kde, color=color, lw=2, label="KDE")
                except:
                    pass
                
                mu, sigma = valid.mean(), valid.std()
                
                mean_val = valid.mean()
                ax.axvline(mean_val, color=MUTED_TEXT, ls='--', lw=2, alpha=0.8)
                
                stats_text = f"n = {n}\nμ = {mean_val:.2f}s\nσ = {sigma:.2f}s"
                ax.annotate(stats_text, xy=(0.03, 0.95), xycoords='axes fraction', 
                            ha='left', va='top', fontsize=TICK_SIZE, color=TEXT_COLOR,
                            bbox=dict(facecolor=BG_COLOR, edgecolor='none', alpha=0.8, pad=0.3))
                            
                if row == 0 and col == 0:
                    ax.plot([], [], color=MUTED_TEXT, ls='--', lw=2, label="Mean")
                    ax.legend(frameon=False, fontsize=TICK_SIZE, loc='upper right')
                
                outliers = valid[(valid < x_min) | (valid > x_max)]
                if len(outliers) > 0:
                    outlier_text = ", ".join([f"{x:.2f}s" for x in outliers.sort_values()])
                    ax.annotate(f"{len(outliers)} solve outside view: {outlier_text}", 
                                xy=(0.98, 0.5), xycoords='axes fraction', ha='right', va='center', 
                                fontsize=TICK_SIZE, color=MUTED_TEXT, 
                                bbox=dict(facecolor=BG_COLOR, edgecolor=GRID_COLOR, alpha=0.8, boxstyle="round,pad=0.3"))
            
            if row == 1:
                ax.set_xlabel("Solve Time (seconds)", fontsize=LABEL_SIZE)
                
            if col == 0:
                ax.set_ylabel("Density", fontsize=LABEL_SIZE)
                
            if row == 0:
                ax.set_title(phase_titles[col], fontsize=TITLE_SIZE, fontweight="bold", pad=15)
                
            if col == 0:
                ax.annotate(player.upper(), xy=(-0.35, 0.5), xycoords='axes fraction',
                            ha='right', va='center', rotation=90, fontsize=TITLE_SIZE, fontweight="bold", color=color)
                            
    axes[0, 0].set_xlim(x_min, x_max)
    fig.suptitle("Solve-Time Distributions", fontsize=TITLE_SIZE + 4, fontweight="bold", x=0.08, y=1.0, ha="left")
    fig.text(0.08, 0.94, "Overall performance vs. before and after the 4–1 comeback boundary", fontsize=LABEL_SIZE, color=MUTED_TEXT, ha="left")
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def generate_comeback_markdown(df, filepath):
    y_b = calculate_player_stats(df, "Yiheng", "Before 4-1")
    y_c = calculate_player_stats(df, "Yiheng", "Comeback")
    
    x_b = calculate_player_stats(df, "Xuanyi", "Before 4-1")
    x_c = calculate_player_stats(df, "Xuanyi", "Comeback")
    
    h_b = calculate_head_to_head_stats(df, "Before 4-1")
    h_c = calculate_head_to_head_stats(df, "Comeback")
    
    def gen_player_table(b, c):
        lines = [
            "| Metric | Before 4–1 | Comeback | Change |",
            "|---|---:|---:|---:|"
        ]
        metrics = [
            ("Valid Solves", "Valid Solves", "{:d}", False, False),
            ("Median (s)", "Median (s)", "{:.2f}", False, False),
            ("Mean (s)", "Mean (s)", "{:.2f}", False, False),
            ("Std Dev (s)", "Std Dev (s)", "{:.2f}", False, False),
            ("IQR (s)", "IQR (s)", "{:.2f}", False, False),
            ("MAD (s)", "MAD (s)", "{:.2f}", False, False),
            ("Best (s)", "Best (s)", "{:.2f}", False, False),
            ("Worst (s)", "Worst (s)", "{:.2f}", False, False),
            ("Race Win %", "Race Win %", "{:.1f}", True, False),
            ("Sub-4s %", "Sub-4.0s %", "{:.1f}", True, False),
            ("5+s %", "5.0+s %", "{:.1f}", True, False),
            ("DNFs", "DNFs", "{:d}", False, False),
            ("+2 Penalties", "+2 Penalties", "{:d}", False, False),
        ]
        
        for name, key, fmt, is_pct, is_margin in metrics:
            val_b = b[key]
            val_c = c[key]
            str_b = format_val(val_b, fmt, is_pct, is_margin)
            str_c = format_val(val_c, fmt, is_pct, is_margin)
            if key in ["Valid Solves", "DNFs", "+2 Penalties"]:
                change = format_change_int(val_b, val_c)
            else:
                change = format_change(val_b, val_c, is_pct)
            lines.append(f"| {name} | {str_b} | {str_c} | {change} |")
        return "\n".join(lines)
        
    md = f"""# Monkey League S6 Grand Final — Comeback Analysis

This report compares performance before and after Xuanyi established a 4–1 set lead.

## Yiheng Wang

{gen_player_table(y_b, y_c)}

## Xuanyi Geng

{gen_player_table(x_b, x_c)}

## Head-to-Head

| Metric | Before 4–1 | Comeback | Change |
|---|---:|---:|---:|
| Median Margin | {format_val(h_b['Median Margin (s)'], '{:.2f}', False, True)} | {format_val(h_c['Median Margin (s)'], '{:.2f}', False, True)} | {format_change(h_b['Median Margin (s)'], h_c['Median Margin (s)'])} |
| Mean Margin | {format_val(h_b['Mean Margin (s)'], '{:.2f}', False, True)} | {format_val(h_c['Mean Margin (s)'], '{:.2f}', False, True)} | {format_change(h_b['Mean Margin (s)'], h_c['Mean Margin (s)'])} |
| Median Absolute Margin | {format_val(h_b['Median |Margin| (s)'], '{:.2f}')} | {format_val(h_c['Median |Margin| (s)'], '{:.2f}')} | {format_change(h_b['Median |Margin| (s)'], h_c['Median |Margin| (s)'])} |

## Key Takeaways

- Yiheng's race-win rate changed from {format_val(y_b['Race Win %'], '{:.1f}', True)} to {format_val(y_c['Race Win %'], '{:.1f}', True)} ({format_change(y_b['Race Win %'], y_c['Race Win %'], True)}).
- Yiheng's median solve changed from {format_val(y_b['Median (s)'], '{:.2f}')}s to {format_val(y_c['Median (s)'], '{:.2f}')}s.
- Yiheng's sub-4 rate changed from {format_val(y_b['Sub-4.0s %'], '{:.1f}', True)} to {format_val(y_c['Sub-4.0s %'], '{:.1f}', True)}.
- Xuanyi's median solve changed from {format_val(x_b['Median (s)'], '{:.2f}')}s to {format_val(x_c['Median (s)'], '{:.2f}')}s.
- Xuanyi's 5+s rate changed from {format_val(x_b['5.0+s %'], '{:.1f}', True)} to {format_val(x_c['5.0+s %'], '{:.1f}', True)}.
- The median head-to-head margin moved from {format_val(h_b['Median Margin (s)'], '{:.2f}', False, True)} to {format_val(h_c['Median Margin (s)'], '{:.2f}', False, True)}.
"""
    with open(filepath, "w") as f:
        f.write(md)

def generate_match_statistics_markdown(df, filepath):
    y_all = calculate_player_stats(df, "Yiheng")
    x_all = calculate_player_stats(df, "Xuanyi")
    y_b = calculate_player_stats(df, "Yiheng", "Before 4-1")
    y_c = calculate_player_stats(df, "Yiheng", "Comeback")
    x_b = calculate_player_stats(df, "Xuanyi", "Before 4-1")
    x_c = calculate_player_stats(df, "Xuanyi", "Comeback")
    h_all = calculate_head_to_head_stats(df)
    
    y_streak_max, y_streak_ranges = calculate_streaks(df, "Yiheng")
    x_streak_max, x_streak_ranges = calculate_streaks(df, "Xuanyi")
    
    y_streak_str = f"{y_streak_max} races (" + ", ".join([f"Races {r[0]}–{r[1]}" for r in y_streak_ranges]) + ")" if y_streak_max > 0 else "0"
    x_streak_str = f"{x_streak_max} races (" + ", ".join([f"Races {r[0]}–{r[1]}" for r in x_streak_ranges]) + ")" if x_streak_max > 0 else "0"

    # Match Overview
    total_races = len(df)
    total_sets = df["Set"].nunique()
    final_score = list(df["match_score_after_set"].values)[-1]
    
    # Timeouts
    timeouts = df[df["Timeout_before"] == 1]
    timeout_lines = []
    for _, row in timeouts.iterrows():
        timeout_lines.append(f"- {row['timeout_by']} timeout: After Race {row['race_number']-1} / Before Race {row['race_number']}")
    timeout_str = "\n".join(timeout_lines) if timeout_lines else "No timeouts."
    
    def format_player_section(name, stats, b_stats, c_stats, streak_str):
        md = f"""## {name}

### Solve Volume
- Total races participated: {stats['Total Races']}
- Valid numerical solves: {stats['Valid Solves']}
- DNFs: {stats['DNFs']}
- +2 penalties: {stats['+2 Penalties']}
- Timeouts taken: {stats['Timeouts Taken']}

### Central Tendency
- Mean solve time: {format_val(stats['Mean (s)'])}s
- Median solve time: {format_val(stats['Median (s)'])}s

### Consistency / Dispersion
- Standard deviation: {format_val(stats['Std Dev (s)'])}s
- IQR: {format_val(stats['IQR (s)'])}s
- MAD: {format_val(stats['MAD (s)'])}s
- Range: {format_val(stats['Best (s)'])}s → {format_val(stats['Worst (s)'])}s

### Extremes
- Best solve: {format_val(stats['Best (s)'])}s (Race {stats['Best Race']})
- Worst valid solve: {format_val(stats['Worst (s)'])}s (Race {stats['Worst Race']})

### Speed Thresholds

| Threshold | Count | Percentage |
|---|---:|---:|
| Sub-3.0s | {stats['Sub-3.0s Count']} | {format_val(stats['Sub-3.0s %'], '{:.1f}', True)} |
| Sub-3.5s | {stats['Sub-3.5s Count']} | {format_val(stats['Sub-3.5s %'], '{:.1f}', True)} |
| Sub-4.0s | {stats['Sub-4.0s Count']} | {format_val(stats['Sub-4.0s %'], '{:.1f}', True)} |
| Sub-4.5s | {stats['Sub-4.5s Count']} | {format_val(stats['Sub-4.5s %'], '{:.1f}', True)} |
| Sub-5.0s | {stats['Sub-5.0s Count']} | {format_val(stats['Sub-5.0s %'], '{:.1f}', True)} |
| 5.0s+ | {stats['5.0+s Count']} | {format_val(stats['5.0+s %'], '{:.1f}', True)} |
| 6.0s+ | {stats['6.0+s Count']} | {format_val(stats['6.0+s %'], '{:.1f}', True)} |
| 7.0s+ | {stats['7.0+s Count']} | {format_val(stats['7.0+s %'], '{:.1f}', True)} |

### Race Performance
- Race wins: {stats['Race Wins']}
- Race losses: {stats['Total Races'] - stats['Race Wins']}
- Race win percentage: {format_val(stats['Race Win %'], '{:.1f}', True)}
- Longest consecutive race-win streak: {streak_str}

### Phase Performance

| Metric | Overall | Before 4–1 | Comeback |
|---|---:|---:|---:|
| Mean (s) | {format_val(stats['Mean (s)'])} | {format_val(b_stats['Mean (s)'])} | {format_val(c_stats['Mean (s)'])} |
| Median (s) | {format_val(stats['Median (s)'])} | {format_val(b_stats['Median (s)'])} | {format_val(c_stats['Median (s)'])} |
| Std Dev (s) | {format_val(stats['Std Dev (s)'])} | {format_val(b_stats['Std Dev (s)'])} | {format_val(c_stats['Std Dev (s)'])} |
| IQR (s) | {format_val(stats['IQR (s)'])} | {format_val(b_stats['IQR (s)'])} | {format_val(c_stats['IQR (s)'])} |
| MAD (s) | {format_val(stats['MAD (s)'])} | {format_val(b_stats['MAD (s)'])} | {format_val(c_stats['MAD (s)'])} |
| Best (s) | {format_val(stats['Best (s)'])} | {format_val(b_stats['Best (s)'])} | {format_val(c_stats['Best (s)'])} |
| Worst (s) | {format_val(stats['Worst (s)'])} | {format_val(b_stats['Worst (s)'])} | {format_val(c_stats['Worst (s)'])} |
| Race Win % | {format_val(stats['Race Win %'], '{:.1f}', True)} | {format_val(b_stats['Race Win %'], '{:.1f}', True)} | {format_val(c_stats['Race Win %'], '{:.1f}', True)} |
| Sub-3.0s % | {format_val(stats['Sub-3.0s %'], '{:.1f}', True)} | {format_val(b_stats['Sub-3.0s %'], '{:.1f}', True)} | {format_val(c_stats['Sub-3.0s %'], '{:.1f}', True)} |
| Sub-3.5s % | {format_val(stats['Sub-3.5s %'], '{:.1f}', True)} | {format_val(b_stats['Sub-3.5s %'], '{:.1f}', True)} | {format_val(c_stats['Sub-3.5s %'], '{:.1f}', True)} |
| Sub-4.0s % | {format_val(stats['Sub-4.0s %'], '{:.1f}', True)} | {format_val(b_stats['Sub-4.0s %'], '{:.1f}', True)} | {format_val(c_stats['Sub-4.0s %'], '{:.1f}', True)} |
| Sub-4.5s % | {format_val(stats['Sub-4.5s %'], '{:.1f}', True)} | {format_val(b_stats['Sub-4.5s %'], '{:.1f}', True)} | {format_val(c_stats['Sub-4.5s %'], '{:.1f}', True)} |
| Sub-5.0s % | {format_val(stats['Sub-5.0s %'], '{:.1f}', True)} | {format_val(b_stats['Sub-5.0s %'], '{:.1f}', True)} | {format_val(c_stats['Sub-5.0s %'], '{:.1f}', True)} |
| 5.0s+ % | {format_val(stats['5.0+s %'], '{:.1f}', True)} | {format_val(b_stats['5.0+s %'], '{:.1f}', True)} | {format_val(c_stats['5.0+s %'], '{:.1f}', True)} |
| DNFs | {stats['DNFs']} | {b_stats['DNFs']} | {c_stats['DNFs']} |
| +2 Penalties | {stats['+2 Penalties']} | {b_stats['+2 Penalties']} | {c_stats['+2 Penalties']} |
"""
        return md
        
    closest_race_str = ""
    if pd.notna(h_all['Closest Race']):
        race_row = df[df["race_number"] == h_all['Closest Race']].iloc[0]
        closest_race_str = f"Race {h_all['Closest Race']} (Yiheng {race_row['Yiheng_final']:.2f}s vs Xuanyi {race_row['Xuanyi_final']:.2f}s, Margin {h_all['Closest |Margin|']:.2f}s)"
        
    largest_race_str = ""
    if pd.notna(h_all['Largest Race']):
        race_row = df[df["race_number"] == h_all['Largest Race']].iloc[0]
        largest_race_str = f"Race {h_all['Largest Race']} (Yiheng {race_row['Yiheng_final']:.2f}s vs Xuanyi {race_row['Xuanyi_final']:.2f}s, Margin {h_all['Largest |Margin|']:.2f}s)"

    md = f"""# Monkey League S6 Grand Final — Match Statistics

## Match Overview
- Total races: {total_races}
- Total sets: {total_sets}
- Final set score: {final_score}
- Total race wins by Yiheng: {y_all['Race Wins']}
- Total race wins by Xuanyi: {x_all['Race Wins']}
- Overall race win % (Yiheng): {format_val(y_all['Race Win %'], '{:.1f}', True)}
- Overall race win % (Xuanyi): {format_val(x_all['Race Win %'], '{:.1f}', True)}
- Timeout count: {len(timeouts)}
- DNF count: {y_all['DNFs'] + x_all['DNFs']}
- +2 penalty count: {y_all['+2 Penalties'] + x_all['+2 Penalties']}

{format_player_section('Yiheng Wang', y_all, y_b, y_c, y_streak_str)}

{format_player_section('Xuanyi Geng', x_all, x_b, x_c, x_streak_str)}

## Head-to-Head Statistics
- Yiheng race wins: {h_all['Yiheng Wins']}
- Xuanyi race wins: {h_all['Xuanyi Wins']}
- Yiheng win percentage: {format_val(h_all['Yiheng Wins'] / total_races * 100, '{:.1f}', True)}
- Xuanyi win percentage: {format_val(h_all['Xuanyi Wins'] / total_races * 100, '{:.1f}', True)}
- Mean signed margin: {format_val(h_all['Mean Margin (s)'], '{:.2f}', False, True)}
- Median signed margin: {format_val(h_all['Median Margin (s)'], '{:.2f}', False, True)}
- Median absolute margin: {format_val(h_all['Median |Margin| (s)'])}
- Mean absolute margin: {format_val(h_all['Mean |Margin| (s)'])}
- Closest race: {closest_race_str}
- Largest valid head-to-head margin: {largest_race_str}

*Note: Positive margins indicate Xuanyi was faster. Negative margins indicate Yiheng was faster.*

## Timeouts
{timeout_str}

## Methodology

### Valid Solve
A solve with a numerical final time. DNFs are excluded from numerical solve-time statistics.

### Final Time
Use the official final solve time. If a +2 penalty exists, the penalized/final time is used.

### Margin
`margin = Yiheng_final - Xuanyi_final`
Positive: Xuanyi faster
Negative: Yiheng faster

### Before 4–1
Sets 1–5.

### Comeback
Sets 6–9.

### Percentage Points
Changes between percentages are expressed in percentage points (pp), not relative percent change.

### Thresholds
Sub-X: time < X
X+: time >= X
"""
    with open(filepath, "w") as f:
        f.write(md)

def run_reports(df, output_dir):
    figures_dir = os.path.join(output_dir, "figures")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    generate_comeback_markdown(df, os.path.join(reports_dir, "comeback_analysis.md"))
    generate_match_statistics_markdown(df, os.path.join(reports_dir, "match_statistics.md"))
    
    plot_solve_distributions(df, os.path.join(figures_dir, "solve_distributions.png"))

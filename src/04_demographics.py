import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, PROJECT_ROOT, participants

OUT = PROJECT_ROOT / "report"
OUT.mkdir(exist_ok=True)

df = participants(DS_REST)

rows = []
for g in ["AD", "FTD", "CN"]:
    s = df[df["group"] == g]
    rows.append({
        "Group": g,
        "n": len(s),
        "Female / Male": f"{(s.Gender == 'F').sum()} / {(s.Gender == 'M').sum()}",
        "Age (mean ± SD)": f"{s.Age.mean():.1f} ± {s.Age.std():.1f}",
        "Age range": f"{s.Age.min()}–{s.Age.max()}",
        "MMSE (mean ± SD)": f"{s.MMSE.mean():.1f} ± {s.MMSE.std():.1f}",
    })
table1 = pd.DataFrame(rows)
print("\n=== Table 1: Participant demographics ===")
print(table1.to_string(index=False))
table1.to_csv(OUT / "table1_demographics.csv", index=False)

ad, ftd, cn = (df[df.group == g] for g in ["AD", "FTD", "CN"])

print("\n=== Confound tests ===")

h, p = stats.kruskal(ad.Age, ftd.Age, cn.Age)
print(f"Age, 3 groups   Kruskal-Wallis  H={h:.3f}  p={p:.4f}")

u, p = stats.mannwhitneyu(ad.Age, cn.Age, alternative="two-sided")
print(f"Age, AD vs CN   Mann-Whitney U  U={u:.1f}  p={p:.4f}")

ct = pd.crosstab(df.group, df.Gender)
chi2, p_gender, dof, _ = stats.chi2_contingency(ct)
print(f"Gender          chi-square      chi2={chi2:.3f}  p={p_gender:.4f}  dof={dof}")
print(ct)

u, p = stats.mannwhitneyu(ad.MMSE, cn.MMSE, alternative="two-sided")
print(f"MMSE, AD vs CN  Mann-Whitney U  U={u:.1f}  p={p:.6f}")
print(f"CN MMSE: min={cn.MMSE.min()} max={cn.MMSE.max()} sd={cn.MMSE.std():.3f}")

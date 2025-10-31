import pandas as pd
from pathlib import Path
p = Path('audit_series.csv')
if not p.exists():
    raise SystemExit('audit_series.csv not found')

df = pd.read_csv(p)
n = len(df)
for tol in [1e-6, 1e-5, 1e-4, 1e-3]:
    act_g = int((df['slack_glu'].abs() <= tol).sum())
    act_z = int((df['slack_xyl'].abs() <= tol).sum())
    print(f"tol={tol:g}: active_glu={act_g}/{n} ({act_g/n*100:.1f}%), active_xyl={act_z}/{n} ({act_z/n*100:.1f}%)")
    print(f"  mean|slack_glu|={df['slack_glu'].abs().mean():.3e}, median={df['slack_glu'].abs().median():.3e}")
    print(f"  mean|slack_xyl|={df['slack_xyl'].abs().mean():.3e}, median={df['slack_xyl'].abs().median():.3e}")

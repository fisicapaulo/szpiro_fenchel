from szpf.experiments import cycle_run
from pathlib import Path
import pandas as pd

def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    rows = []
    for n in [6, 8, 10, 12, 16]:
        out = cycle_run(n=n, renorm=True, rcond=1e-10)
        rows.append(dict(n=n, kappa=out["kappa"], resid=out["resid"]))
    df = pd.DataFrame(rows)
    save_table(df, "results/tables/cycle_summary.csv")
    print(df)

if __name__ == "__main__":
    main()

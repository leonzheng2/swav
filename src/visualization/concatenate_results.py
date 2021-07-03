import pandas
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Concatenate csv results into one csv')
parser.add_argument("results_dir")
args = parser.parse_args()

all_results = []
results_dir = Path(args.results_dir)
for csv_file in results_dir.iterdir():
    if csv_file.stem.isdecimal():
        print(f"Reading {csv_file}")
        all_results.append(pandas.read_csv(csv_file, index_col='id'))

print(all_results)
print("Save the concatenated csv results file")
results_df = pandas.concat(all_results)
results_df.to_csv(results_dir / "results.csv")

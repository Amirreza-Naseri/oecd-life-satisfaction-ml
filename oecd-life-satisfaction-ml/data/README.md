# Data

The raw OECD Better Life Index dataset is intentionally not committed to this repository.

## Recommended reproducible setup

From the repository root:

```bash
python src/download_data.py
```

This downloads the frozen historical `DF_BLI` mirror used to match the original notebook and saves it as:

```text
data/oecd_bli.csv
```

To try the canonical OECD archive endpoint instead:

```bash
python src/download_data.py --source oecd
```

The training loader accepts either of the common value-column names:

- `Value`
- `OBS_VALUE`

and filters total-population rows using either:

- `Inequality == "Total"`
- `INEQUALITY == "TOT"`

The project expects at least `Country`, `INDICATOR`, and `Indicator` plus one of the value and inequality column variants above.

from __future__ import annotations

import argparse
from pathlib import Path

import requests

# Public mirror of the OECD DF_BLI export that matches the dataset used by the
# original notebook (target values and country coverage verified against it).
MIRROR_URL = (
    "https://gist.githubusercontent.com/suo/21371d187e8d97c347610f9a57837630/"
    "raw/fa7cdcfa6531d7a3dc2da2be2e96dcdbedd8bc11/gistfile1.txt"
)

# Canonical archived OECD dataflow. The mirror is used by default because it
# freezes the historical snapshot for reproducibility.
OECD_ARCHIVE_URL = (
    "https://sdmx.oecd.org/archive/rest/data/OECD,DF_BLI,/all"
    "?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)


def download(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text
    if "INDICATOR" not in text or "Country" not in text:
        raise RuntimeError("Downloaded content does not look like OECD BLI CSV data.")
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OECD Better Life Index data.")
    parser.add_argument("--output", type=Path, default=Path("data/oecd_bli.csv"))
    parser.add_argument(
        "--source",
        choices=["mirror", "oecd"],
        default="mirror",
        help="Use the frozen public mirror for reproducibility or the OECD archive endpoint.",
    )
    args = parser.parse_args()
    url = MIRROR_URL if args.source == "mirror" else OECD_ARCHIVE_URL
    path = download(url, args.output)
    print(f"Saved dataset to: {path.resolve()}")


if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import datetime

def load_fasta_or_txt(path: str) -> str:
    txt = Path(path).read_text().strip()
    if txt.startswith(">"):
        return "".join(
            line.strip() for line in txt.splitlines()
            if line and not line.startswith(">")
        ).upper()
    return txt.upper()

def write_fasta(records, out_path):
    with open(out_path, "w") as fh:
        for rid, seq in records:
            fh.write(f">{rid}\n{seq}\n")

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

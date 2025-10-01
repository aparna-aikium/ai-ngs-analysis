IUPAC_DNA = {
    "A":"A","C":"C","G":"G","T":"T",
    "R":"AG","Y":"CT","S":"GC","W":"AT","K":"GT","M":"AC",
    "B":"CGT","D":"AGT","H":"ACT","V":"ACG","N":"ACGT"
}
AA20 = set("ARNDCEQGHILKMFPSTWYV")

def normalize_degeneracy(seq_type: str, chars: str) -> str:
    chars = chars.upper().replace(" ", "")
    if seq_type.lower() == "dna":
        expanded = []
        for ch in chars:
            if ch in IUPAC_DNA:
                expanded.extend(IUPAC_DNA[ch])
            elif ch in "ACGT":
                expanded.append(ch)
            else:
                raise ValueError(f"Invalid DNA char: {ch}")
        return "".join(sorted(set(expanded)))
    bad = set(chars) - AA20
    if bad:
        raise ValueError(f"Invalid AA chars: {bad}")
    return "".join(sorted(set(chars)))

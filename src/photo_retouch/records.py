"""Write-once result tables: the first execution saves them, later ones must match."""
import pandas as pd

TOLERANCE = 1e-6


def compare_with_record(new_table, saved_table):
    """Rows compared, the largest numeric difference, and text cells that differ."""
    # Methods saved in more than one experiment appear twice; keep the latest row.
    saved_table = saved_table[~saved_table.index.duplicated(keep="last")]
    shared = new_table.index.intersection(saved_table.index)
    new = new_table.loc[shared, saved_table.columns]
    old = saved_table.loc[shared]
    numeric = old.select_dtypes("number").columns
    text = old.columns.difference(numeric)
    return {
        "rows compared": len(shared),
        "max numeric difference": (
            new[numeric].astype(float) - old[numeric]
        ).abs().max().max(),
        "text cells differing": int((new[text].astype(str) != old[text]).sum().sum()),
    }


def save_or_check(table, path, tolerance=TOLERANCE):
    """Save ``table`` to ``path``; if the file exists, the table must reproduce it."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path)
        return "saved"
    saved_table = pd.read_csv(path, index_col=list(range(table.index.nlevels)))
    check = compare_with_record(table, saved_table)
    assert check["rows compared"] == len(table), (path.name, check)
    assert check["max numeric difference"] < tolerance, (path.name, check)
    assert check["text cells differing"] == 0, (path.name, check)
    return "reproduced"

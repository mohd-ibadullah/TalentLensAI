"""Convert the submission CSV to XLSX format for portal upload."""
import csv
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    from openpyxl import Workbook

def csv_to_xlsx(csv_path: str, xlsx_path: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Ranked Candidates"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            ws.append(row)

    # Auto-width columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)

    wb.save(xlsx_path)
    print(f"Saved XLSX: {xlsx_path} ({ws.max_row - 1} candidates)")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "outputs" / "mohd_ibadullah.csv"
    xlsx_path = root / "outputs" / "mohd_ibadullah.xlsx"
    csv_to_xlsx(str(csv_path), str(xlsx_path))

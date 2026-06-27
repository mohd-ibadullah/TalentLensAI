"""Update phone in submission_metadata.yaml before portal upload.
Usage: python scripts/set_phone.py "+91 98765 43210"
   or: set SUBMISSION_PHONE=+919876543210 && python scripts/set_phone.py
"""
import os
import re
import sys
from pathlib import Path

META = Path(__file__).resolve().parent.parent / "submission_metadata.yaml"


def main() -> None:
    phone = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SUBMISSION_PHONE", "")).strip()
    if not phone or "99999" in phone:
        print("ERROR: Provide a real phone number.")
        print("  python scripts/set_phone.py \"+91 98765 43210\"")
        sys.exit(1)
    text = META.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'phone:\s*"[^"]*"',
        f'phone: "{phone}"',
        text,
        count=1,
    )
    if n != 1:
        print("ERROR: Could not update phone field in submission_metadata.yaml")
        sys.exit(1)
    META.write_text(new_text, encoding="utf-8")
    print(f"Updated {META} with phone: {phone}")


if __name__ == "__main__":
    main()

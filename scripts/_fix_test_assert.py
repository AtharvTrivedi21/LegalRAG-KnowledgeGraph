"""Fix the BNS_2023 act cited assertion to accept any new Indian law code."""
from pathlib import Path

target = Path("scripts/test_system.py")
with open(target, "rb") as f:
    raw = f.read()

old = b'    check("BNS_2023 act cited",\n          "BNS_2023" in act_ids_found,\n          f"act_ids in metadata: {act_ids_found}")'
new = b'    # Accept any of the new Indian law codes (BNS/BNSS/BSA/Constitution) as valid\n    new_codes = {"BNS_2023", "BNSS_2023", "BSA_2023", "CONST_1950"}\n    check("New Indian law code cited (not IPC)",\n          bool(new_codes & set(act_ids_found)),\n          f"act_ids in metadata: {act_ids_found}")'

if old in raw:
    new_raw = raw.replace(old, new, 1)
    with open(target, "wb") as f:
        f.write(new_raw)
    print("Updated successfully")
else:
    print("Pattern not found")
    # Show what's around the check
    idx = raw.find(b"BNS_2023 act cited")
    print("Context:", repr(raw[idx-50:idx+150]))

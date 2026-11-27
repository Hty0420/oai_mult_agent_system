from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\data\oa\test225")

df = pd.read_csv(ROOT / "manifest.csv")

windows = [
    "le30days",
    "le60days",
    "le90days"
]

rows = []

for _, row in df.iterrows():

    patient_id = str(row["SubjectID"])

    for window in windows:

        folder = ROOT / window / patient_id

        xray = folder / "xray_bilateral.png"
        mri = folder / "mri_right.nii.gz"
        mask = folder / "mask_right.nii.gz"

        if (
            xray.is_file()
            and mri.is_file()
            and mask.is_file()
        ):
            item = row.to_dict()

            item["XRayPath"] = str(xray)
            item["ImagePath"] = str(mri)
            item["MaskPath"] = str(mask)

            item["TimeWindow"] = window

            rows.append(item)
            break

result = pd.DataFrame(rows)

result.to_csv(
    ROOT / "manifest_216.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Available multimodal cases:", len(result))
print("Unique patients:", result["SubjectID"].nunique())
print("Time window distribution:")
print(result["TimeWindow"].value_counts())
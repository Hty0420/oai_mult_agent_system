from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\data\oa\test225")

manifest = pd.read_csv(ROOT / "manifest.csv")

WINDOWS = ["le30days", "le60days", "le90days"]

def find_patient_dir(patient_id):
    patient_id = str(patient_id)

    for window in WINDOWS:
        patient_dir = ROOT / window / patient_id

        if patient_dir.is_dir():
            return patient_dir, window

    return None, None


results = []

for _, row in manifest.iterrows():
    patient_id = str(row["SubjectID"])

    patient_dir, window = find_patient_dir(patient_id)

    if patient_dir is None:
        results.append({
            "SubjectID": patient_id,
            "window": None,
            "xray_exists": False,
            "mri_exists": False,
            "mask_exists": False
        })
        continue

    results.append({
        "SubjectID": patient_id,
        "window": window,
        "xray_exists": (
            patient_dir / "xray_bilateral.png"
        ).is_file(),
        "mri_exists": (
            patient_dir / "mri_right.nii.gz"
        ).is_file(),
        "mask_exists": (
            patient_dir / "mask_right.nii.gz"
        ).is_file()
    })

result_df = pd.DataFrame(results)

print("\nData check results:")
print("Total cases:", len(result_df))
print("X-ray:", result_df["xray_exists"].sum())
print("MRI:", result_df["mri_exists"].sum())
print("Mask:", result_df["mask_exists"].sum())

print("\nFound in:")
print(result_df["window"].value_counts(dropna=False))

result_df.to_csv(
    ROOT / "image_path_check.csv",
    index=False
)
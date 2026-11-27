"""Audit field and actual local file availability for an OAI data root.
Usage: python check_dataset.py --data-dir "D:\\data\\oa\\test225"
"""
import argparse
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--data-dir', required=True)
args=ap.parse_args()
root=Path(args.data_dir)
manifest=pd.read_csv(root/'manifest.csv',dtype={'SubjectID':str})
print(f'Cases: {len(manifest)}; unique patients: {manifest.SubjectID.nunique()}')
print('KL distribution:',manifest.KLGrade.value_counts().sort_index().to_dict())
print('Time difference:',manifest.MRMinusXRayDays.describe().to_dict())
for path_field, default_name in [('XRayPath','xray_bilateral.png'),('ImagePath','mri_right.nii.gz'),('MaskPath','mask_right.nii.gz')]:
    present=[]
    for _,r in manifest.iterrows():
        original=Path(str(r[path_field]))
        local=root/str(r['SubjectID'])/default_name
        present.append(original.is_file() or local.is_file())
    print(f'{path_field} present: {sum(present)}/{len(present)}')
print('Clinical visit counts:',manifest.clinical_visit.value_counts(dropna=False).to_dict() if 'clinical_visit' in manifest else 'not supplied')

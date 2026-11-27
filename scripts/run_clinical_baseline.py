"""First REAL (clinical-only) five-class baseline for the OAI manifest.
Usage: python run_clinical_baseline.py --data-dir "D:\\data\\oa\\test225" --output-dir outputs/clinical_baseline
Only metadata is needed. No image-model claims are made.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CLASSES = np.arange(5)


def multiclass_brier(y, probs):
    one_hot = np.eye(5)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece_top_label(y, probs, n_bins=5):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y))
    boundaries = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for k in range(n_bins):
        in_bin = (confidence >= boundaries[k]) & ((confidence <= boundaries[k+1]) if k == n_bins-1 else (confidence < boundaries[k+1]))
        if in_bin.any():
            result += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(result)


def evaluate(y, p):
    pred = p.argmax(axis=1)
    return {
        'n': len(y),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
        'qwk': float(cohen_kappa_score(y, pred, labels=CLASSES, weights='quadratic')),
        'brier_sum_5class': multiclass_brier(y, p),
        'nll': float(log_loss(y, np.clip(p, 1e-12, 1), labels=CLASSES)),
        'ece_top_label_5bins': ece_top_label(y, p),
        'class_counts': {str(k): int((np.asarray(y)==k).sum()) for k in CLASSES}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='Folder containing manifest.csv/train.csv/test.csv')
    ap.add_argument('--output-dir', default='outputs/clinical_baseline')
    args = ap.parse_args()
    root = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / 'train.csv', dtype={'SubjectID':str})
    test = pd.read_csv(root / 'test.csv', dtype={'SubjectID':str})
    required=['SubjectID','KLGrade','Age','BMI','GenderOriginalCode','BMIMissing']
    for title, df in [('train',train),('test',test)]:
        missing=set(required)-set(df.columns)
        if missing: raise ValueError(f'{title} missing columns: {missing}')
        if not df['KLGrade'].isin(CLASSES).all(): raise ValueError(f'{title}: KL labels must be 0..4')
        if df['SubjectID'].duplicated().any(): raise ValueError(f'{title}: duplicate patients')
    overlap=set(train.SubjectID)&set(test.SubjectID)
    if overlap: raise ValueError(f'Patient leakage: {len(overlap)} subjects appear in both splits')
    if len(set(train.KLGrade))<5: raise ValueError('Training split lacks some KL classes')
    for df in (train,test):
        df['GenderOriginalCode']=df['GenderOriginalCode'].astype('string').fillna('unknown')
    columns=['Age','BMI','BMIMissing','GenderOriginalCode']
    preprocess=ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]), ['Age','BMI','BMIMissing']),
        ('category', OneHotEncoder(handle_unknown='ignore'), ['GenderOriginalCode']),
    ])
    model=Pipeline([('preprocess',preprocess),('classifier',LogisticRegression(max_iter=2000,random_state=42))])
    model.fit(train[columns],train.KLGrade.astype(int))
    p=model.predict_proba(test[columns])
    # Align columns to global KL 0..4 even if implementation changes.
    aligned=np.zeros((len(test),5), dtype=float)
    aligned[:,model.named_steps['classifier'].classes_.astype(int)] = p
    metrics=evaluate(test.KLGrade.to_numpy(dtype=int),aligned)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    predictions=test[['SubjectID','KLGrade']].copy()
    predictions['pred_kl']=aligned.argmax(axis=1)
    for k in range(5): predictions[f'prob_kl{k}']=aligned[:,k]
    predictions['predictive_entropy']=-np.sum(np.where(aligned>0,aligned*np.log(np.maximum(aligned,1e-12)),0.0),axis=1)
    predictions.to_csv(out/'clinical_predictions.csv',index=False,encoding='utf-8-sig')
    print('Clinical-only baseline finished. This is NOT an X-ray/MRI or Agent experiment.')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print(f'Wrote: {out.resolve()}')
    print('NOTE: Clinical visit provenance is not independently confirmed in the supplied metadata.')
    print('NOTE: Test set is small; do not tune models based on these test metrics.')

if __name__=='__main__':
    main()

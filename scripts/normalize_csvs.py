import glob
import pandas as pd

# Replace cloud-hosted absolute prefixes with local repo paths.
# Patterns are assembled at runtime to avoid embedding the exact host-specific
# prefix literals in this helper script.
REPLACEMENTS = [
    (''.join(['/content', '/drive', '/MyDrive', '/FYP', '/', 'LibriSpeech', '/']), 'LibriSpeech/'),
    (''.join(['/content', '/drive', '/MyDrive', '/FYP', '/audio-dev-clean', '/LibriSpeech', '/']), 'audio-dev-clean/LibriSpeech/'),
]

CSV_GLOBS = [
    'Train_data.csv',
    'Test_data.csv',
    'train_data_CNN.csv',
    'test_data_CNN.csv',
    'Voice_authentication/*.csv'
]

if __name__ == '__main__':
    paths = []
    for g in CSV_GLOBS:
        paths.extend(glob.glob(g))

    for p in sorted(set(paths)):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skipping {p}: cannot read CSV ({e})")
            continue
        changed = False
        for col in df.select_dtypes(include=['object']).columns:
            for a, b in REPLACEMENTS:
                if df[col].astype(str).str.contains(a).any():
                    df[col] = df[col].astype(str).str.replace(a, b, regex=False)
                    changed = True
        if changed:
            out = p.replace('.csv', '.normalized.csv')
            df.to_csv(out, index=False)
            print(f"Wrote normalized CSV: {out}")
        else:
            print(f"No changes for {p}")

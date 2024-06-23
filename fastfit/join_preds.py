import os
import json
import pandas as pd

l = sorted([f for f in os.listdir('.') if 'preds_' in f])
dfs = [pd.read_json(f, lines=True) for f in l]

dfs[0]['vectors'] = dfs[0].apply(lambda row: [
    row['embedding'],
    dfs[1].loc[row.name, 'embedding'],
    dfs[2].loc[row.name, 'embedding'],
    dfs[3].loc[row.name, 'embedding']
], axis=1)

dfs[0].to_json('embedding_history.jsonl', lines=True, orient="records")

#! MIT License
#!
#! Copyright (c) 2025 Santos O. G., Helen C. S. C. Lima,
#! Permission is hereby granted, free of charge, to any person obtaining a copy
#! of this software and associated documentation files (the "Software"), to deal
#! in the Software without restriction, including without limitation the rights
#! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#! copies of the Software, and to permit persons to whom the Software is
#! furnished to do so, subject to the following conditions:
#!
#! The above copyright notice and this permission notice shall be included in all
#! copies or substantial portions of the Software.

import argparse
import re
import pandas as pd
from arff import load, dump
from sklearn.preprocessing import KBinsDiscretizer

def merge_classes(df, class_col, min_count=10):
    df = df.copy()
    df[class_col] = df[class_col].astype(str)
    while True:
        counts = df[class_col].value_counts()
        to_merge = counts[counts < min_count].index.tolist()
        if not to_merge:
            break
        for cls in to_merge:
            cls_clean = cls.lstrip('R')
            parts = cls_clean.split('.')
            if len(parts) < 2:
                continue
            parent = 'R' + '.'.join(parts[:-1])
            df.loc[df[class_col] == cls, class_col] = parent
    # Remove prefix 'R' e normalize strings
    df[class_col] = df[class_col].apply(lambda x: re.sub(r'^R\.?', '', str(x)))
    return df

def discretize_df(df, exclude_cols=None, max_bins=20):
    """
    Apply Equal Frequency Binning (quantile) for each numeric attribute,
    with max_bins partitions. If not possible, use the highest number of
    intervals (>1).
    """
    df = df.copy()
    exclude_cols = exclude_cols or []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            unique = df[col].nunique()
            bins = min(max_bins, unique)
            for b in range(bins, 1, -1):
                try:
                    est = KBinsDiscretizer(
                        n_bins=b,
                        encode='ordinal',
                        strategy='quantile',
                        quantile_method='averaged_inverted_cdf'
                    )
                    df[col] = est.fit_transform(df[[col]]).astype(int).flatten()
                    break
                except Exception:
                    continue
    return df

def arff_to_df(input_file):
    with open(input_file, 'r') as f:
        arff_data = load(f)
    attrs = arff_data['attributes']
    df = pd.DataFrame(
        arff_data['data'],
        columns=[a[0] for a in attrs],  # type: ignore[arg-type]
    )

    types = {a[0]: a[1] for a in attrs}
    return df, arff_data['relation'], types

def df_to_arff(df, relation, types, output_file):
    attributes = []
    for col in df.columns:
        t = types.get(col)
        if t == 'numeric':
            attributes.append((col, 'NUMERIC'))
        else:
            # Para atributos nominais, remove valores vazios e '?'
            vals = [v for v in sorted(df[col].astype(str).unique()) if v not in ('', '?')]
            attributes.append((col, vals))
    arff_dict = {
        'relation': relation,
        'attributes': attributes,
        'data': df.values.tolist()
    }

    for att in arff_dict['attributes']:
        if att[0] != "class":
            attr_vec = att[1]
            for i in range(len(attr_vec)):
                attr_vec[i] = str(int(float(attr_vec[i])))
    for data_arr in arff_dict['data']:
        for num in range(len(data_arr)- 1):
            data_arr[num] = str(int(float(data_arr[num])))

    with open(output_file, 'w') as f:
        dump(arff_dict, f)


def main():
    parser = argparse.ArgumentParser(description='Merge small classes, clean and discretize ARFF.')
    parser.add_argument('input_file', help='path to ARFF file')
    args = parser.parse_args()

    df, relation, types = arff_to_df(args.input_file)
    class_col = df.columns[-1]

    df = merge_classes(df, class_col)

    # remove lines with '?' (without value after merge)
    df = df[df[class_col] != '?']

    # discretization
    df = discretize_df(df, exclude_cols=[class_col])

    # save result
    output = args.input_file.replace('.arff', '_discretized.arff')
    df_to_arff(df, relation, types, output)
    print(f'Saved discretized ARFF to {output}')

if __name__ == '__main__':
    main()

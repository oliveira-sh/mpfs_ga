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

def str2int(value: str) -> int:
    """
    Extracts all digit characters from a string and returns them as an integer.
    If no digits are found, returns 0.
    Fast-path for pure-digit strings.
    """
    if value.isdigit():
        return int(value)
    digits = [c for c in value if c.isdigit()]
    return int(''.join(digits)) if digits else 0

def get_datasets_profile(training_file: str, test_file: str):
    """
    Reads ARFF-style files and returns (n_train, n_test, n_attributes).
    """
    n_train = 0
    last_line = None
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip().lower().startswith('@data'):
                break
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            n_train += 1
            last_line = s
    if last_line is None:
        raise ValueError(f"No data instances found in training file: {training_file}")

    n_attributes = len(last_line.split(','))
    n_test = 0
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip().lower().startswith('@data'):
                break
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            n_test += 1

    return n_train, n_test, n_attributes

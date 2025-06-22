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

import os
import random
from io import StringIO
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import numpy as np
from classifier import Classifier
from charge_training_set import ChargeTrainingSet
from charge_test_set import ChargeTestSet

# --- globals for workers ---
HEADER = NAMES = RECS = FOLDS = MLNP = USF = None

# --- vectorized GA operators ---
def evolve_population(pop, scores, cxpb, mutpb):
    pop_size, n_feats = pop.shape
    a = np.random.randint(0, pop_size, size=pop_size)
    b = np.random.randint(0, pop_size, size=pop_size)
    a_scores = np.take(scores, a)
    b_scores = np.take(scores, b)
    parents = np.where(
        a_scores[:, None] >= b_scores[:, None],
        pop[a],
        pop[b]
    )

    # one-point crossover
    np.random.shuffle(parents)
    offspring = parents.copy()
    do_cx = np.random.rand(pop_size // 2) < cxpb
    cuts = np.random.randint(1, n_feats, size=pop_size // 2)
    for i, (apply_cx, cx) in enumerate(zip(do_cx, cuts)):
        if not apply_cx:
            continue
        i1, i2 = 2*i, 2*i+1
        tmp = offspring[i1, cx:].copy()
        offspring[i1, cx:], offspring[i2, cx:] = offspring[i2, cx:], tmp

    # bit-flip mutation
    mut_mask = np.random.rand(pop_size, n_feats) < mutpb
    offspring ^= mut_mask

    return offspring

def init_worker(header, names, recs, folds, mlnp, usf):
    """Initialize globals in each worker once."""
    global HEADER, NAMES, RECS, FOLDS, MLNP, USF
    HEADER, NAMES, RECS, FOLDS = header, names, recs, folds
    MLNP, USF = mlnp, usf

def fitness_worker(mask):
    """Called in each worker process; mask is a list of bools."""
    return fitness_in_memory(mask)

def parse_arff(path):
    header, data = [], []
    with open(path,'r') as f:
        in_data = False
        for line in f:
            if not in_data and line.strip().lower().startswith('@data'):
                in_data = True
                continue
            if in_data:
                if line.strip() and not line.strip().startswith('%'):
                    data.append(line.strip().split(','))
            else:
                header.append(line)
    attrs = [l for l in header if l.strip().lower().startswith('@attribute')]
    names = [a.split()[1] for a in attrs]
    return header, names, data

def build_arff_text(header, names, records, mask):
    cls = names[-1]
    # filter header
    filtered = []
    cnt = 0
    for line in header:
        if line.strip().lower().startswith('@attribute'):
            nm = line.split()[1]
            if nm == cls:
                filtered.append(line)
            else:
                if mask[cnt]:
                    filtered.append(line)
                cnt += 1
        else:
            filtered.append(line)

    # build data lines
    feat_idx = [i for i,keep in enumerate(mask) if keep]
    lines = [
        ','.join([rec[i] for i in feat_idx] + [rec[-1]])
        for rec in records
    ]

    return (
        ''.join(filtered)
        + '\n@data\n'
        + '\n'.join(lines)
        + '\n'
    )

def fitness_in_memory(mask):
    """
    For a given boolean mask, perform 5-fold CV on RECS and return mean hF.
    Uses HEADER, NAMES, RECS, FOLDS, MLNP, USF from globals.
    """
    header, names, recs, folds = HEADER, NAMES, RECS, FOLDS
    scores = []

    if recs == None or folds == None or MLNP == None or USF == None:
        raise Exception("Some globalVar = None")

    for train_idx, valid_idx in folds:
        train_recs = [recs[i] for i in train_idx]
        valid_recs = [recs[i] for i in valid_idx]

        # build ARFF texts
        train_txt = build_arff_text(header, names, train_recs, mask)
        valid_txt = build_arff_text(header, names, valid_recs, mask)

        n_attr = sum(mask) + 1
        # train‐set
        ctr = ChargeTrainingSet(None, n_attr, len(train_recs), MLNP)
        ctr.open_training_file  = lambda: setattr(ctr, '_fin', StringIO(train_txt))
        ctr.close_training_file = lambda: setattr(ctr, '_fin', None)
        ctr.get_training_set()

        # validation‐set (using ChargeTestSet API)
        cte = ChargeTestSet(None, len(valid_recs), n_attr)
        cte.open_test_file  = lambda: setattr(cte, '_fin', StringIO(valid_txt))
        cte.close_test_file = lambda: setattr(cte, '_fin', None)
        cte.get_test_set()

        # classify & score
        cl = Classifier(len(train_recs), len(valid_recs), n_attr, "", USF)
        Classifier.auxCLCTR = ctr
        Classifier.auxCLCTE = cte
        scores.append(cl.apply_classifier(False))

    return sum(scores) / len(scores)

def main():
    p = ArgumentParser()
    p.add_argument('--train', required=True,
                   help="ARFF used for 5-fold CV")
    p.add_argument('--pop',   type=int,   default=20)
    p.add_argument('--gen',   type=int,   default=40)
    p.add_argument('--cxpb',  type=float, default=0.7)
    p.add_argument('--mutpb', type=float, default=0.2)
    p.add_argument('--mlnp',  action='store_false',
                   help="flag mandatory leaf nodes")
    p.add_argument('--usf',   action='store_true',
                   help="use log-usefulness")
    p.add_argument('--out',   type=str,   default='out_ga',
                   help="pasta de saída para ARFF final")
    args = p.parse_args()

    # 1. load full ARFF
    header, names, recs = parse_arff(args.train)
    n = len(recs)
    n_feats = len(names) - 1

    # 2. build 5 folds (shuffle once, deterministic seed)
    indices = list(range(n))
    random.seed(0)
    random.shuffle(indices)
    fold_size = n // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n
        valid = indices[start:end]
        train = [idx for idx in indices if idx not in valid]
        folds.append((train, valid))

    # 3) initialize population as boolean matrix
    pop = np.random.rand(args.pop, n_feats) < 0.5
    best_mask: list[bool] = []
    best_score = -1.0

    # 4) start a single Pool with globals
    chunksize = max(1, args.pop // (cpu_count() * 4))
    with Pool(initializer=init_worker,
              initargs=(header, names, recs, folds, args.mlnp, args.usf)) as pool:

        for gen in range(1, args.gen + 1):
            # evaluate fitness in parallel
            masks = pop.tolist()
            scores = pool.map(fitness_worker, masks, chunksize)

            # track best
            i_best = max(range(len(scores)), key=lambda i: scores[i])
            if scores[i_best] > best_score:
                best_score = scores[i_best]
                best_mask  = pop[i_best].tolist()

            print(f"Gen {gen:2d}: max = {max(scores):.4f}, avg = {sum(scores)/len(scores):.4f}")
            pop = evolve_population(pop, scores, args.cxpb, args.mutpb)

    # 5) write final filtered ARFF using best_mask
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'train_opt.arff')
    with open(out_path, 'w') as f:
        f.write(build_arff_text(header, names, recs, best_mask))

    print(f"\nBest CV-hF = {best_score:.4f} with {sum(best_mask)}/{n_feats} attributes.")
    print(f"Final ARFF in {out_path}")

if __name__ == '__main__':
    main()

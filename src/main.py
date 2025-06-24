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

from classifier import Classifier
from charge_training_set import ChargeTrainingSet
from charge_test_set import ChargeTestSet

# globals for worker processes
globals_: tuple = (None, None, None, None, None, None)
HEADER, NAMES, RECS, FOLDS, MLNP, USF = globals_

def evolve_population(pop, scores, cxpb, mutpb):
    pop_size = len(pop)
    n_feats = len(pop[0])
    # tournament selection
    parents = []
    for _ in range(pop_size):
        i, j = random.randrange(pop_size), random.randrange(pop_size)
        parents.append(pop[i] if scores[i] >= scores[j] else pop[j])

    # one-point crossover
    offspring = [p.copy() for p in parents]
    random.shuffle(offspring)
    for i in range(0, pop_size - pop_size % 2, 2):
        if random.random() < cxpb:
            cut = random.randint(1, n_feats - 1)
            offspring[i][cut:], offspring[i+1][cut:] = \
                offspring[i+1][cut:], offspring[i][cut:]

    # bit-flip mutation
    for individual in offspring:
        for idx in range(n_feats):
            if random.random() < mutpb:
                individual[idx] = not individual[idx]
    return offspring

def init_worker(header, names, recs, folds, mlnp, usf):
    global HEADER, NAMES, RECS, FOLDS, MLNP, USF
    HEADER, NAMES, RECS, FOLDS, MLNP, USF = header, names, recs, folds, mlnp, usf

def fitness_worker(mask):
    return fitness_in_memory(mask)

def parse_arff(path):
    header, data = [], []
    with open(path, 'r') as f:
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
    filtered = []
    cnt = 0
    for line in header:
        if line.strip().lower().startswith('@attribute'):
            nm = line.split()[1]
            if nm == cls or mask[cnt]:
                filtered.append(line)
            if nm != cls:
                cnt += 1
        else:
            filtered.append(line)

    feat_idx = [i for i, keep in enumerate(mask) if keep]
    lines = [
        ','.join([rec[i] for i in feat_idx] + [rec[-1]])
        for rec in records
    ]

    return ''.join(filtered) + '\n@data\n' + '\n'.join(lines) + '\n'

def fitness_in_memory(mask):
    header, names, recs, folds = HEADER, NAMES, RECS, FOLDS
    scores = []
    if None in (recs, folds, MLNP, USF):
        raise Exception("Worker globals not set")

    for train_idx, valid_idx in folds:
        train_recs = [recs[i] for i in train_idx]
        valid_recs = [recs[i] for i in valid_idx]

        train_txt = build_arff_text(header, names, train_recs, mask)
        valid_txt = build_arff_text(header, names, valid_recs, mask)

        n_attr = sum(mask) + 1
        ctr = ChargeTrainingSet(None, n_attr, len(train_recs), MLNP)
        ctr.open_training_file = lambda: setattr(ctr, '_fin', StringIO(train_txt))
        ctr.close_training_file = lambda: setattr(ctr, '_fin', None)
        ctr.get_training_set()

        cte = ChargeTestSet(None, len(valid_recs), n_attr)
        cte.open_test_file = lambda: setattr(cte, '_fin', StringIO(valid_txt))
        cte.close_test_file = lambda: setattr(cte, '_fin', None)
        cte.get_test_set()

        cl = Classifier(len(train_recs), len(valid_recs), n_attr, "", USF)
        Classifier.auxCLCTR = ctr
        Classifier.auxCLCTE = cte
        scores.append(cl.apply_classifier(False))

    return sum(scores) / len(scores)

def main():
    p = ArgumentParser()
    p.add_argument('--train', required=True, help="ARFF used for 5-fold CV")
    p.add_argument('--pop',   type=int,   default=20)
    p.add_argument('--gen',   type=int,   default=40)
    p.add_argument('--cxpb',  type=float, default=0.7)
    p.add_argument('--mutpb', type=float, default=0.2)
    p.add_argument('--mlnp',  action='store_false', help="flag mandatory leaf nodes")
    p.add_argument('--usf',   action='store_true',  help="use log-usefulness")
    p.add_argument('--out',   type=str,   default='out_ga', help="output folder for final ARFF")
    args = p.parse_args()

    header, names, recs = parse_arff(args.train)
    n = len(recs)
    n_feats = len(names) - 1

    indices = list(range(n))
    random.seed(0)
    random.shuffle(indices)
    fold_size = n // 5
    folds = []
    for i in range(5):
        start, end = i * fold_size, n if i == 4 else (i + 1) * fold_size
        valid = indices[start:end]
        train = [idx for idx in indices if idx not in valid]
        folds.append((train, valid))

    pop = [[random.random() < 0.5 for _ in range(n_feats)] for _ in range(args.pop)]
    best_mask, best_score = [], -1.0

    chunksize = max(1, args.pop // (cpu_count() * 4))
    print("gen\tmax\tavg")
    with Pool(initializer=init_worker,
              initargs=(header, names, recs, folds, args.mlnp, args.usf)) as pool:
        for gen in range(1, args.gen + 1):
            masks = pop
            scores = pool.map(fitness_worker, masks, chunksize)

            i_best = max(range(len(scores)), key=lambda i: scores[i])
            if scores[i_best] > best_score:
                best_score = scores[i_best]
                best_mask  = pop[i_best].copy()

            print(f"{gen:2d}\t{max(scores):.4f}\t{sum(scores)/len(scores):.4f}")
            pop = evolve_population(pop, scores, args.cxpb, args.mutpb)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'train_opt.arff')
    with open(out_path, 'w') as f:
        f.write(build_arff_text(header, names, recs, best_mask))
    print(f"\nBest = {best_score:.4f} with {sum(best_mask)}/{n_feats} attributes in {out_path}.")

if __name__ == '__main__':
    main()

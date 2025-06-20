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

import sys
import math
from utils import get_datasets_profile
from charge_training_set import ChargeTrainingSet
from charge_test_set import ChargeTestSet

class Classifier:
    # Will be set by nbayes()
    auxCLCTR: ChargeTrainingSet
    auxCLCTE: ChargeTestSet

    def __init__(
        self,
        number_of_training_examples: int,
        number_of_test_examples: int,
        number_of_attributes: int,
        result_file: str,
        usefulness: bool,
    ):
        self.number_of_training_examples = number_of_training_examples
        self.number_of_test_examples = number_of_test_examples
        self.number_of_attributes = number_of_attributes
        self.result_file = result_file
        self.usefulness = usefulness
        self.fout = None

        # Flags & storage for precomputed tables
        self._prepared = False
        self._classes = []
        self._log_priors = {}
        self._log_usefulness = {}
        self._attr_log = {}

    def open_result_file(self):
        try:
            self.fout = open(self.result_file, 'w')
        except IOError:
            print(f"Error opening result file {self.result_file}")
            sys.exit(1)

    def close_result_file(self):
        if self.fout:
            self.fout.close()
            self.fout = None

    def _prepare_log_probabilities(self):
        """Compute once: log-priors, log-usefulness, and log-likelihoods."""
        if self._prepared:
            return

        ctr = Classifier.auxCLCTR
        n_train = self.number_of_training_examples

        # 1) Classes to evaluate
        classes = list(ctr.classes_for_probability_evaluation.keys())
        self._classes = classes

        # 2) Log-priors
        lp = {}
        for cid in classes:
            freq = ctr.get_class_frequency(cid)
            lp[cid] = math.log10(freq / n_train) if freq > 0 else float('-inf')
        self._log_priors = lp

        # 3) Log-usefulness (if enabled)
        if self.usefulness:
            lu = {}
            for cid, uval in ctr.classes_for_probability_evaluation.items():
                lu[cid] = math.log10(uval) if uval > 0 else float('-inf')
            self._log_usefulness = lu

        # 4) Per-class, per-attribute, per-value log-likelihoods
        smoothing_log = math.log10(1 / n_train) # Pre-calculate
        attr_idx = ctr.attribute_index
        num_attrs = self.number_of_attributes - 1

        alog = {}
        for cid in classes:
            freq_list = ctr.class_freq[cid]
            cfreq = ctr.get_class_frequency(cid)
            if cfreq == 0:
                # If class frequency is zero, all likelihoods for this class should be smoothing_log
                class_attr_logs = []
                for aid in range(num_attrs):
                    start = attr_idx[aid]
                    end = attr_idx[aid + 1]
                    max_val = end - start
                    logs = [smoothing_log] * (max_val + 1)
                    class_attr_logs.append(logs)
                alog[cid] = class_attr_logs
                continue

            class_attr_logs = []
            for aid in range(num_attrs):
                start = attr_idx[aid]
                end = attr_idx[aid + 1]
                max_val = end - start
                logs = [0.0] * (max_val + 1)
                for v in range(max_val + 1):
                    f = freq_list[start + v]
                    logs[v] = math.log10(f / cfreq) if f > 0 else smoothing_log
                class_attr_logs.append(logs)
            alog[cid] = class_attr_logs

        self._attr_log = alog
        self._prepared = True

    def apply_classifier(self, use_stdout: bool) -> float:
            self._prepare_log_probabilities()
            if use_stdout:
                self.open_result_file()

            self._split_cache = {}
            cte = Classifier.auxCLCTE
            test_set = cte.test_set
            true_classes = cte.class_test_set

            classes = self._classes
            lp = self._log_priors
            lu = self._log_usefulness if self.usefulness else None
            alog = self._attr_log

            n_train = self.number_of_training_examples
            num_attrs = self.number_of_attributes - 1

            # OPTIMIZATION: Calculate smoothing value once outside loop
            smoothing_log = math.log10(1 / n_train) if n_train > 0 else float('-inf')
            numerator = sumP = sumT = sumMinPT = 0

            true_classes_split = [tc.split('.') for tc in true_classes]
            for ex_idx, feat in enumerate(test_set):
                best_score = float('-inf')
                best_class = ""

                for cid in classes:
                    score = lp[cid]
                    if self.usefulness and lu is not None:
                        score += lu[cid]

                    calogs = alog.get(cid)
                    if calogs is None:
                        continue

                    for aid in range(num_attrs):
                        val = feat[aid]
                        logs = calogs[aid]
                        score += logs[val] if val < len(logs) else smoothing_log

                    if score > best_score:
                        best_score = score
                        best_class = cid
                if best_class in self._split_cache:
                    best_class_parts = self._split_cache[best_class]
                else:
                    best_class_parts = best_class.split('.')
                    self._split_cache[best_class] = best_class_parts

                true_c_parts = true_classes_split[ex_idx]

                inter = 0
                for a, b in zip(true_c_parts, best_class_parts):
                    if a == b:
                        inter += 1
                    else:
                        break
                numerator += inter

                sizeP = len(best_class_parts)
                sizeT = len(true_c_parts)
                sumP += sizeP
                sumT += sizeT
                sumMinPT += min(sizeP, sizeT)

                if use_stdout and self.fout is not None:
                    self.fout.write(f"Example {ex_idx} ({'.'.join(true_c_parts)}) -> {best_class}\n")
            hP_new = numerator / sumMinPT if sumMinPT > 0 else 0.0
            hP = numerator / sumP if sumP > 0 else 0.0
            hR = numerator / sumT if sumT > 0 else 0.0

            if (hP + hR) == 0:
                hF = 0.0
            else:
                hF = 100 * (2 * hP * hR) / (hP + hR)
            result = hF

            if use_stdout and self.fout is not None:
                self.fout.write(f"hP = {hP_new * 100}\n")
                self.fout.write(f"hR = {hR * 100}\n")
                self.fout.write(f"hF = {hF}\n")
                self.close_result_file()
            return result

def nbayes(
    mlnp: bool,
    usf: bool,
    training_file: str,
    test_file: str,
    result_file: str,
) -> float:
    # 1) Profile datasets
    n_train, n_test, n_attr = get_datasets_profile(training_file, test_file)
    # 2) Load and index training
    ctr = ChargeTrainingSet(training_file, n_attr, n_train, mlnp)
    ctr.get_training_set()
    # 3) Load test set
    cte = ChargeTestSet(test_file, n_test, n_attr)
    cte.get_test_set()
    # 4) Classifier
    cl = Classifier(n_train, n_test, n_attr, result_file, usf)
    Classifier.auxCLCTR = ctr
    Classifier.auxCLCTE = cte
    return cl.apply_classifier(bool(result_file))

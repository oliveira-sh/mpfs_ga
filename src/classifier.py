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
    # Static references to training and test sets
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

    def compute_probability_attribute_class(
        self,
        exte: int,
        attribute_id: int,
        class_id: str
    ) -> float:
        freq = self.auxCLCTR.get_attribute_value_class_frequency(
            attribute_id,
            self.auxCLCTE.get_attribute_value_test_set(exte, attribute_id),
            class_id,
        )
        class_freq = self.auxCLCTR.get_class_frequency(class_id)
        return freq / class_freq

    def compute_probability_training_class(self, class_id: str) -> float:
        return self.auxCLCTR.get_class_frequency(class_id) / self.number_of_training_examples

    @staticmethod
    def split_class(class_str: str) -> list[str]:
        return class_str.split('.')

    @staticmethod
    def intersection_pt(true_class: str, predicted_class: str) -> int:
        t = Classifier.split_class(true_class)
        p = Classifier.split_class(predicted_class)
        count = 0
        for a, b in zip(t, p):
            if a == b:
                count += 1
            else:
                break
        return count

    def apply_classifier(self, use_stdout: bool) -> float:
        numerator = 0
        sumP = 0
        sumT = 0
        sumMinPT = 0

        if use_stdout:
            self.open_result_file()

        for exte in range(self.number_of_test_examples):
            bestP = float('-inf')
            best_class: str = ""

            # Evaluate each candidate class
            for class_id, usefulness_value in self.auxCLCTR.classes_for_probability_evaluation.items():
                p_log = 0.0
                # Attribute probabilities
                for attr_id in range(self.number_of_attributes - 1):
                    temp = self.compute_probability_attribute_class(exte, attr_id, class_id)
                    if temp == 0.0:
                        temp = 1.0 / self.number_of_training_examples
                    p_log += math.log10(temp)
                # Prior probability
                p_log += math.log10(self.compute_probability_training_class(class_id))

                # Convert back from log and apply usefulness if required
                if self.usefulness:
                    p_val = math.pow(10, p_log) * usefulness_value
                else:
                    p_val = math.pow(10, p_log)

                if p_val > bestP:
                    bestP = p_val
                    best_class = class_id

            true_class = self.auxCLCTE.get_class_test_set(exte)
            intersection = self.intersection_pt(true_class, best_class)
            numerator += intersection

            pred_segments = Classifier.split_class(best_class)
            true_segments = Classifier.split_class(true_class)
            sizeP = len(pred_segments)
            sizeT = len(true_segments)
            sumP += sizeP
            sumT += sizeT
            sumMinPT += min(sizeP, sizeT)

            if use_stdout and self.fout is not None:
                self.fout.write(f"Example {exte} ({true_class}) -> {best_class}\n")

        # Harmonic metrics
        hP_new = numerator / sumMinPT
        hP = numerator / sumP
        hR = numerator / sumT
        result = 100 * (2 * hP * hR) / (hP + hR)

        if use_stdout and self.fout is not None:
            self.fout.write(f"hP = {hP_new * 100}\n")
            self.fout.write(f"hR = {hR * 100}\n")
            self.fout.write(f"hF = {100 * (2 * hP_new * hR) / (hP_new + hR)}\n")
            self.close_result_file()

        return result


def nbayes(
    mlnp: bool,
    usf: bool,
    training_file: str,
    test_file: str,
    save: bool,
) -> float:
    # Retrieve dataset profiles
    n_train, n_test, n_attr = get_datasets_profile(training_file, test_file)

    # Prepare training and test sets
    ctr = ChargeTrainingSet(training_file, n_attr, n_train, mlnp)
    ctr.get_training_set()
    cte = ChargeTestSet(test_file, n_test, n_attr)
    cte.get_test_set()

    # Initialize classifier
    result_file = 'out.txt' if save else ''

    cl = Classifier(n_train, n_test, n_attr, result_file, usf)
    Classifier.auxCLCTR = ctr
    Classifier.auxCLCTE = cte

    return cl.apply_classifier(save)

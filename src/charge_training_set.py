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
from typing import Dict, List
from utils import str2int


class ChargeTrainingSet:
    def __init__(
        self,
        training_file: str,
        number_of_attributes: int,
        number_of_training_examples: int,
        mandatory_leaf_node_prediction: bool,
    ):
        """
        :param training_file: Path to the ARFF-style training file.
        :param number_of_attributes: Total number of attributes (including class label).
        :param number_of_training_examples: Expected number of training instances.
        :param mandatory_leaf_node_prediction: Whether to only predict leaf classes.
        """
        self.training_file = training_file
        self.number_of_attributes = number_of_attributes
        self.number_of_training_examples = number_of_training_examples
        self.mandatory_leaf_node_prediction = mandatory_leaf_node_prediction

        # Cumulative offsets for attribute values
        self.attribute_index: List[int] = [0] * (self.number_of_attributes + 1)
        self.number_of_independent_attribute_values: int = 0
        self.biggest_level: int = 0

        # Frequency of attribute-value occurrences per class
        self.class_freq: Dict[str, List[int]] = {}
        self.counter: List[int] = []

        # Structures for computing class usefulness
        self.classes_for_probability_evaluation: Dict[str, float] = {}
        self.class_per_level: List[str] = []

        # File handle
        self._fin = None

    def open_training_file(self):
        try:
            self._fin = open(self.training_file, 'r')
        except IOError:
            sys.stderr.write(f"[ERR] Error opening training file {self.training_file}\n")
            sys.exit(1)

    def close_training_file(self):
        if self._fin:
            self._fin.close()
            self._fin = None

    def update_attribute_index(self, index: int, number_of_values: int):
        """Sets cumulative offset for attribute at position index."""
        self.attribute_index[index] = self.attribute_index[index - 1] + number_of_values

    def compute_number_of_independent_attribute_values(self) -> int:
        """Returns total discrete values across all independent attributes."""
        return self.attribute_index[self.number_of_attributes - 1]

    def get_attribute_index(self, attribute_id: int) -> int:
        """Gets the cumulative offset for a given attribute."""
        return self.attribute_index[attribute_id]

    def initialize_counter(self):
        """Resets and initializes the per-class counter vector."""
        size = self.number_of_independent_attribute_values + 1
        self.counter = [0] * size
        # Last position holds total examples count for class
        self.counter[self.number_of_independent_attribute_values] = 1

    def update_counter(self, index: int):
        """Increments the counter at the given global index."""
        self.counter[index] += 1

    def update_class_freq(self, class_id: str):
        """
        Updates frequency table for a class based on the current counter vector.
        Initializes on first encounter, otherwise accumulates.
        """
        if class_id not in self.class_freq:
            self.class_freq[class_id] = self.counter.copy()
        else:
            existing = self.class_freq[class_id]
            for i, cnt in enumerate(self.counter):
                if cnt == 1:
                    existing[i] += 1

    def get_attribute_value_class_frequency(
        self, attribute_id: int, attribute_value: int, class_id: str
    ) -> int:
        """Fetches frequency of a specific attribute value for a class."""
        if class_id in self.class_freq:
            idx = self.get_attribute_index(attribute_id) + attribute_value
            return self.class_freq[class_id][idx]
        return 0

    def get_class_frequency(self, class_id: str) -> int:
        """Fetches total count of examples for a given class."""
        if class_id in self.class_freq:
            return self.class_freq[class_id][self.number_of_independent_attribute_values]
        return 0

    def not_exist_child(self, class_id: str) -> bool:
        """Checks if the given class has no subclasses in the frequency table."""
        prefix = class_id + '.'
        return not any(key.startswith(prefix) for key in self.class_freq)

    def set_classes_for_probability_evaluation(self, class_id: str):
        """
        Registers classes (and optionally ancestors) for usefulness computation.
        For mandatory leaf-node prediction, only leaf classes are added.
        """
        if class_id in self.classes_for_probability_evaluation:
            return
        if self.mandatory_leaf_node_prediction:
            if self.not_exist_child(class_id):
                self.classes_for_probability_evaluation[class_id] = 1.0
                # Add ancestor paths
                begin = -1
                father = ''
                while True:
                    end = class_id.find('.', begin + 1)
                    if end == -1:
                        break
                    part = class_id[begin+1:end]
                    father += part
                    if father in self.classes_for_probability_evaluation:
                        self.classes_for_probability_evaluation[father] = 1.0
                    begin = end
                    father += '.'
        else:
            self.classes_for_probability_evaluation[class_id] = 1.0
            begin = -1
            father = ''
            while True:
                end = class_id.find('.', begin + 1)
                if end == -1:
                    break
                part = class_id[begin+1:end]
                father += part
                self.classes_for_probability_evaluation[father] = 1.0
                begin = end
                father += '.'

    def compute_class_usefulness(self):
        """
        Computes usefulness of each registered class based on hierarchy depth.
        Uses logarithmic scaling relative to maximum tree size.
        """
        # Reset all usefulness counters
        for cls in self.classes_for_probability_evaluation:
            self.classes_for_probability_evaluation[cls] = 1.0
        # Count hierarchy contributions
        for cls in list(self.classes_for_probability_evaluation):
            begin = -1
            father = ''
            while True:
                end = cls.find('.', begin + 1)
                if end == -1:
                    break
                part = cls[begin+1:end]
                father += part
                if father in self.classes_for_probability_evaluation:
                    self.classes_for_probability_evaluation[father] += 1.0
                begin = end
                father += '.'
        # Determine maximum count
        max_count = max(self.classes_for_probability_evaluation.values(), default=1.0)
        # Final usefulness scaling
        for cls, count in self.classes_for_probability_evaluation.items():
            self.classes_for_probability_evaluation[cls] = (
                1 - math.log2(count) / math.log2(max_count + 1)
            )

    def initialize_class_per_level(self, class_id: str) -> int:
        """Splits a class ID by levels (separated by '.') and stores segments."""
        self.class_per_level = []
        begin = -1
        while True:
            end = class_id.find('.', begin + 1)
            if end == -1:
                break
            self.class_per_level.append(class_id[begin+1:end])
            begin = end
        # Last segment
        self.class_per_level.append(class_id[begin+1:])
        return len(self.class_per_level)

    def get_class_per_level(self, index: int) -> str:
        """Returns the class segment at the specified hierarchy level."""
        return self.class_per_level[index]

    def get_subset_class(self, level: int) -> str:
        """Constructs the class path up to the given level."""
        subset = ''
        for i in range(level):
            subset += self.get_class_per_level(i) + '.'
        return subset[:-1]

    def get_training_set(self):
        """
        Parses the ARFF training file to build attribute indices,
        initializes class evaluation structures, and accumulates
        frequencies from the data section.
        """
        self.open_training_file()

        if self._fin is None:
            exit()

        # Header parsing: attribute definitions
        last_line = ''
        attribute_id = 0
        for raw in self._fin:
            line = raw.strip()
            lower = line.lower()
            if lower.startswith('@data'):
                break
            if not lower.startswith('@attribute'):
                continue
            # Count discrete values in braces
            number_of_values = 1
            begin = line.find('{', 9)
            while True:
                end = line.find(',', begin + 1)
                if end == -1:
                    break
                number_of_values += 1
                begin = end
            self.update_attribute_index(attribute_id + 1, number_of_values)
            attribute_id += 1
            last_line = line

        # Total discrete attribute values
        self.number_of_independent_attribute_values = (
            self.compute_number_of_independent_attribute_values()
        )

        # Initialize classes from header enumeration
        begin = last_line.find('{', 9)
        while True:
            end = last_line.find(',', begin + 1)
            if end == -1:
                break
            cls = last_line[begin+1:end]
            self.set_classes_for_probability_evaluation(cls)
            begin = end
        # Last class in braces
        end = last_line.find('}', begin + 1)
        cls = last_line[begin+1:end]
        self.set_classes_for_probability_evaluation(cls)

        # Data section parsing
        for raw in self._fin:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            self.initialize_counter()
            # Extract class label
            stop = line.rfind(',')
            class_id = line[stop+1:]
            # Parse each attribute value
            begin = -1
            attribute_id = 0
            while begin < stop:
                end = line.find(',', begin + 1)
                value_str = line[begin+1:end]
                val = str2int(value_str)
                # Validate and update counter
                max_val = (
                    self.get_attribute_index(attribute_id + 1)
                    - self.get_attribute_index(attribute_id)
                )
                if val <= max_val:
                    self.update_counter(self.get_attribute_index(attribute_id) + val)
                else:
                    sys.stderr.write(
                        f"[ERR] Inconsistent Attribute Value for AttributeId: {attribute_id}\n"
                    )
                    sys.exit(1)
                begin = end
                attribute_id += 1
            # Update class frequency per hierarchical subset
            levels = self.initialize_class_per_level(class_id)
            if levels > self.biggest_level:
                self.biggest_level = levels
            for lvl in range(1, levels + 1):
                subset = self.get_subset_class(lvl)
                self.update_class_freq(subset)

        # Final usefulness computation
        self.compute_class_usefulness()
        self.close_training_file()

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
from utils import str2int
from typing import Dict, List, Optional

class ChargeTrainingSet:
    def __init__(
        self,
        training_file: Optional[str],
        number_of_attributes: int,
        number_of_training_examples: int,
        mandatory_leaf_node_prediction: bool,
    ):
        self.training_file = training_file
        self.number_of_attributes = number_of_attributes
        self.number_of_training_examples = number_of_training_examples
        self.mandatory_leaf_node_prediction = mandatory_leaf_node_prediction

        self.attribute_index: List[int] = [0] * (self.number_of_attributes + 1)
        self.number_of_independent_attribute_values: int = 0
        self.biggest_level: int = 0

        self.class_freq: Dict[str, List[int]] = {}
        self.counter: List[int] = []

        self.classes_for_probability_evaluation: Dict[str, float] = {}
        self.class_per_level: List[str] = []

        self._fin = None

    def open_training_file(self):
        try:
            self._fin = open(str(self.training_file), 'r')
        except IOError:
            sys.stderr.write(f"[ERR] Error opening training file {self.training_file}\n")
            sys.exit(1)

    def close_training_file(self):
        if self._fin:
            self._fin.close()
            self._fin = None

    def update_attribute_index(self, index: int, number_of_values: int):
        self.attribute_index[index] = self.attribute_index[index - 1] + number_of_values

    def compute_number_of_independent_attribute_values(self) -> int:
        return self.attribute_index[self.number_of_attributes - 1]

    def get_attribute_index(self, attribute_id: int) -> int:
        return self.attribute_index[attribute_id]

    def initialize_counter(self):
        size = self.number_of_independent_attribute_values + 1
        self.counter = [0] * size
        self.counter[self.number_of_independent_attribute_values] = 1

    def update_counter(self, index: int):
        self.counter[index] += 1

    def update_class_freq(self, class_id: str):
        if class_id not in self.class_freq:
            self.class_freq[class_id] = self.counter[:] # Use slice for shallow copy
        else:
            existing = self.class_freq[class_id]
            for i, cnt in enumerate(self.counter):
                if cnt == 1: # Only increment if the attribute was present in the current example
                    existing[i] += 1

    def get_attribute_value_class_frequency(
        self, attribute_id: int, attribute_value: int, class_id: str
    ) -> int:
        if class_id in self.class_freq:
            idx = self.get_attribute_index(attribute_id) + attribute_value
            return self.class_freq[class_id][idx]
        return 0

    def get_class_frequency(self, class_id: str) -> int:
        if class_id in self.class_freq:
            return self.class_freq[class_id][self.number_of_independent_attribute_values]
        return 0

    def not_exist_child(self, class_id: str) -> bool:
        prefix = class_id + '.'
        return not any(key.startswith(prefix) for key in self.class_freq)

    def set_classes_for_probability_evaluation(self, class_id: str):
        if class_id in self.classes_for_probability_evaluation:
            return
        if self.mandatory_leaf_node_prediction:
            if self.not_exist_child(class_id):
                self.classes_for_probability_evaluation[class_id] = 1.0
                # add ancestors...
                parts = class_id.split('.')
                father_parts = []
                for part in parts[:-1]: # Iterate up to the parent, excluding the leaf itself
                    father_parts.append(part)
                    father = '.'.join(father_parts)
                    if father not in self.classes_for_probability_evaluation:
                        self.classes_for_probability_evaluation[father] = 1.0
        else:
            # register all levels
            self.classes_for_probability_evaluation[class_id] = 1.0
            parts = class_id.split('.')
            father_parts = []
            for part in parts[:-1]: # Iterate up to the parent, excluding the leaf itself
                father_parts.append(part)
                father = '.'.join(father_parts)
                self.classes_for_probability_evaluation[father] = 1.0


    def compute_class_usefulness(self):
        for cls in self.classes_for_probability_evaluation:
            self.classes_for_probability_evaluation[cls] = 1.0 # Reset counts to 1

        # Optimize: iterate over items and update counts in one pass
        items_to_process = list(self.classes_for_probability_evaluation.keys())
        for cls in items_to_process:
            parts = cls.split('.')
            father_parts = []
            for part in parts[:-1]:
                father_parts.append(part)
                father = '.'.join(father_parts)
                if father in self.classes_for_probability_evaluation:
                    self.classes_for_probability_evaluation[father] += 1.0

        max_count = max(self.classes_for_probability_evaluation.values(), default=1.0)
        log2_max_count_plus_1 = math.log2(max_count + 1)
        if log2_max_count_plus_1 == 0: # Avoid division by zero if max_count is 0
            for cls in self.classes_for_probability_evaluation:
                self.classes_for_probability_evaluation[cls] = 0.0
        else:
            for cls, count in self.classes_for_probability_evaluation.items():
                self.classes_for_probability_evaluation[cls] = (
                    1 - math.log2(count) / log2_max_count_plus_1
                )

    def initialize_class_per_level(self, class_id: str) -> int:
        self.class_per_level = class_id.split('.')
        return len(self.class_per_level)

    def get_class_per_level(self, index: int) -> str:
        return self.class_per_level[index]

    def get_subset_class(self, level: int) -> str:
        return '.'.join(self.class_per_level[:level])

    def get_training_set(self):
        self.open_training_file()
        if self._fin is None:
            return

        # 1) Parse @attribute lines, build attribute_index, remember last attribute line
        last_attr_line = self._parse_header()

        # 2) From that last @attribute line, extract and register all class labels
        self._initialize_classes(last_attr_line)

        # 3) Read the data section, updating counters and class frequencies
        self._parse_data_section()

        # 4) Finalize
        self.compute_class_usefulness()
        self.close_training_file()

    def _parse_header(self) -> str:
        """
        Reads lines until '@data', registering each '@attribute' for its number of values.
        Returns the last '@attribute' line (which defines the classes).
        """
        if self._fin is None:
            raise Exception("self._fin is None");
        last_line = ''
        attribute_id = 0

        for raw in self._fin:
            line = raw.strip()
            if line.lower().startswith('@data'):
                break
            if not line.lower().startswith('@attribute'):
                continue

            # count comma-separated values inside { }
            # Optimized to use find and count directly, avoiding a loop for each line
            begin_brace = line.find('{')
            if begin_brace != -1:
                end_brace = line.rfind('}')
                if end_brace != -1:
                    values_str = line[begin_brace + 1 : end_brace]
                    number_of_values = values_str.count(',') + 1
                else: # Fallback if '}' is not found, should not happen in valid ARFF
                    number_of_values = 1
            else: # No braces found, assume single value (e.g., numeric attribute)
                number_of_values = 1
            self.update_attribute_index(attribute_id + 1, number_of_values)
            attribute_id += 1
            last_line = line
        self.number_of_independent_attribute_values = (
            self.compute_number_of_independent_attribute_values()
        )
        return last_line

    def _initialize_classes(self, class_line: str):
        """
        Given the final '@attribute … {A,B,C}' line, register each class label.
        """
        # parse comma-separated tokens inside { }
        begin = class_line.find('{')
        end = class_line.rfind('}')
        if begin != -1 and end != -1:
            class_values_str = class_line[begin + 1 : end]
            for cls in class_values_str.split(','):
                self.set_classes_for_probability_evaluation(cls.strip()) # .strip() to remove whitespace

    def _parse_data_section(self):
        """
        Iterates over each data line, updating attribute counters and class frequencies.
        """
        if self._fin is None:
            raise Exception("self._fin is None");

        # Pre-fetch attribute index values for faster access within the loop
        attribute_indices = self.attribute_index
        update_counter_func = self.update_counter
        update_class_freq_func = self.update_class_freq
        initialize_class_per_level_func = self.initialize_class_per_level
        get_subset_class_func = self.get_subset_class
        biggest_level = self.biggest_level

        for raw in self._fin:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue

            # reset counts for this example
            self.initialize_counter()

            # split off the class label at the last comma
            stop = line.rfind(',')
            class_id = line[stop + 1 :]
            attributes_str = line[:stop]

            # Optimized: process attributes in one go
            parts = attributes_str.split(',')
            for attribute_id, raw_val in enumerate(parts):
                val = str2int(raw_val)
                max_val = (
                    attribute_indices[attribute_id + 1]
                    - attribute_indices[attribute_id]
                )
                if val <= max_val:
                    idx = attribute_indices[attribute_id] + val
                    update_counter_func(idx)
                else:
                    sys.stderr.write(
                        f"[ERR] Inconsistent Attribute Value for AttributeId,RawVal: {attribute_id},{raw_val}\n"
                    )
                    sys.exit(1)

            # update hierarchical class-frequency
            levels = initialize_class_per_level_func(class_id)
            if levels > biggest_level:
                biggest_level = levels
            for lvl in range(1, levels + 1):
                subset = get_subset_class_func(lvl)
                update_class_freq_func(subset)
        self.biggest_level = biggest_level

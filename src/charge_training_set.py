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
            self._fin = open(self.training_file, 'r')
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
            self.class_freq[class_id] = self.counter.copy()
        else:
            existing = self.class_freq[class_id]
            for i, cnt in enumerate(self.counter):
                if cnt == 1:
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
            # register all levels
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
        for cls in self.classes_for_probability_evaluation:
            self.classes_for_probability_evaluation[cls] = 1.0
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
        max_count = max(self.classes_for_probability_evaluation.values(), default=1.0)
        for cls, count in self.classes_for_probability_evaluation.items():
            self.classes_for_probability_evaluation[cls] = (
                1 - math.log2(count) / math.log2(max_count + 1)
            )

    def initialize_class_per_level(self, class_id: str) -> int:
        self.class_per_level = []
        begin = -1
        while True:
            end = class_id.find('.', begin + 1)
            if end == -1:
                break
            self.class_per_level.append(class_id[begin+1:end])
            begin = end
        self.class_per_level.append(class_id[begin+1:])
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

        # after header we know how many total attribute values there are:
        self.number_of_independent_attribute_values = (
            self.compute_number_of_independent_attribute_values()
        )
        return last_line


    def _initialize_classes(self, class_line: str):
        """
        Given the final '@attribute … {A,B,C}' line, register each class label.
        """
        # parse comma-separated tokens inside { }
        begin = class_line.find('{', 9)
        while True:
            end = class_line.find(',', begin + 1)
            if end == -1:
                break
            cls = class_line[begin + 1 : end]
            self.set_classes_for_probability_evaluation(cls)
            begin = end

        # last one before the closing '}'
        end = class_line.find('}', begin + 1)
        cls = class_line[begin + 1 : end]
        self.set_classes_for_probability_evaluation(cls)


    def _parse_data_section(self):
        """
        Iterates over each data line, updating attribute counters and class frequencies.
        """
        if self._fin is None:
            raise Exception("self._fin is None");

        for raw in self._fin:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue

            # reset counts for this example
            self.initialize_counter()

            # split off the class label at the last comma
            stop = line.rfind(',')
            class_id = line[stop + 1 :]

            # walk through each attribute value
            begin = -1
            attribute_id = 0
            while begin < stop:
                end = line.find(',', begin + 1)
                val = str2int(line[begin + 1 : end])

                # check against our registered domain size
                max_val = (
                    self.get_attribute_index(attribute_id + 1)
                    - self.get_attribute_index(attribute_id)
                )
                if val <= max_val:
                    idx = self.get_attribute_index(attribute_id) + val
                    self.update_counter(idx)
                else:
                    sys.stderr.write(
                        f"[ERR] Inconsistent Attribute Value for AttributeId: {attribute_id}\n"
                    )
                    sys.exit(1)

                begin = end
                attribute_id += 1

            # update hierarchical class-frequency
            levels = self.initialize_class_per_level(class_id)
            if levels > self.biggest_level:
                self.biggest_level = levels
            for lvl in range(1, levels + 1):
                subset = self.get_subset_class(lvl)
                self.update_class_freq(subset)

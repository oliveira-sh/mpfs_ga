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

from utils import str2int
import sys

class ChargeTestSet:
    def __init__(self, test_file: str, number_of_test_examples: int, number_of_attributes: int):
        """
        Initializes the test set container.
        :param test_file: Path to the ARFF-style test file.
        :param number_of_test_examples: Expected number of test instances.
        :param number_of_attributes: Total attributes per instance (including class label).
        """
        self.test_file = test_file
        self.number_of_test_examples = number_of_test_examples
        self.number_of_attributes = number_of_attributes

        # allocate a 2D list for attribute values (exclude the class label)
        self.test_set = [
            [0] * (self.number_of_attributes - 1)
            for _ in range(self.number_of_test_examples)
        ]
        # allocate a list for class labels
        self.class_test_set = [''] * self.number_of_test_examples
        self._fin = None

    def open_test_file(self):
        """Opens the test file, exits on failure."""
        try:
            self._fin = open(self.test_file, 'r')
        except IOError:
            sys.stderr.write(f"[ERR] Error opening test file {self.test_file}\n")
            sys.exit(1)

    def close_test_file(self):
        """Closes the test file."""
        if self._fin:
            self._fin.close()
            self._fin = None

    def insert_test_set(self, example_id: int, attribute_id: int, attribute_value: int):
        """Inserts a single attribute value into the test_set matrix."""
        self.test_set[example_id][attribute_id] = attribute_value

    def get_attribute_value_test_set(self, example_id: int, attribute_id: int) -> int:
        """Retrieves an attribute value from the test_set matrix."""
        return self.test_set[example_id][attribute_id]

    def insert_class_test_set(self, example_id: int, class_value: str):
        """Inserts a class label for a given example index."""
        self.class_test_set[example_id] = class_value

    def get_class_test_set(self, example_id: int) -> str:
        """Retrieves the class label for a given example index."""
        return self.class_test_set[example_id]

    def get_test_set(self):
        """
        Reads the ARFF 'data' section and populates test_set and class_test_set.
        Skips comments (%) and blank lines. Uses lowercase and str2int from utils.
        """
        self.open_test_file()
        if self._fin is None:
            exit()

        # skip header until we hit '@data'
        for line in self._fin:
            if (line.strip()[:5]).lower() == '@data':
                break

        example_id = 0
        # read each data line
        for line in self._fin:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            parts = line.split(',')
            # parse attribute values
            for attr_id, raw_val in enumerate(parts[:-1]):
                self.test_set[example_id][attr_id] = str2int(raw_val)
            # parse class label
            self.class_test_set[example_id] = parts[-1]
            example_id += 1
        self.close_test_file()

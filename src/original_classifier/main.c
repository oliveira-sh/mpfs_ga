//! MIT License
//!
//! Copyright (c) 2025 Santos O. G., Helen C. S. C. Lima,
//! Copyright (c) 2009 Jr, C. N. S. Freitas
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.

#include <stdio.h>
#include <stdbool.h>
#include "classifier/c_nbayes.h"

int main() {
    long double result = nbayes_c(false, false,
                                  "train.arff",
                                  "test.arff",
                                  "out.txt");

    printf("Resultado: %.10Lf\n", result);

    return 0;
}

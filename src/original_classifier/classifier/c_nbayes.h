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

#ifndef C_NBAYES_H
#define C_NBAYES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/// Global Model Naive Bayes Classifier - C Interface
///
/// This function provides a C-compatible interface for the Naive Bayes classifier.
/// It performs classification on test data using a model trained from the training data.
///
/// @param mlnp Mandatory Leaf Node Prediction flag
///             - true: Enables mandatory leaf node prediction
///             - false: Disables mandatory leaf node prediction
///
/// @param usf Usefulness flag
///            - true: Enables usefulness calculations/optimizations
///            - false: Disables usefulness calculations/optimizations
///
/// @param trainingFile Path to the training dataset file
///                     Must be a valid file path containing the training data
///                     in the expected format (e.g., ARFF format)
///
/// @param testFile Path to the test dataset file
///                 Must be a valid file path containing the test data
///                 in the same format as the training file
///
/// @param resultFile Path to the output results file
///                   - Pass a valid file path to save results to file
///                   - Pass empty string ("") to output results to stdout only
///                   - Results include classification accuracy and predictions
///
/// @return Classification accuracy as a percentage (0.0 to 100.0)
///         Returns the overall accuracy of the classifier on the test set
///
/// @note This is a C wrapper function that converts C-style strings to C++ strings
///       and calls the internal nbayes() function.
///
/// @warning Ensure all file paths are valid and accessible before calling this function.
///          Invalid file paths will cause the program to terminate with an error.
///
///
/// @example
/// // Example usage with result file output
/// long double accuracy = nbayes_c(true, false,
///                                "data/train.arff",
///                                "data/test.arff",
///                                "results/output.txt");
///
///
/// // Example usage with stdout output only
/// long double accuracy = nbayes_c(false, true,
///                                "data/train.arff",
///                                "data/test.arff",
///                                "");
long double nbayes_c(bool mlnp, bool usf,
                     const char* trainingFile, const char* testFile,
                     const char* resultFile);

#ifdef __cplusplus
}
#endif

#endif // C_NBAYES_H

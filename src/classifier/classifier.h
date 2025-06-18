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

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <limits>

#include "chargeTrainingSet.h"
#include "chargeTestSet.h"
#include "utils.h"

class Classifier {
public:
    static ChargeTrainingSet* auxCLCTR;
    static ChargeTestSet* auxCLCTE;

    ifstream fin;
    ofstream fout;

    unsigned int numberOfTrainingExamples;
    unsigned int numberOfTestExamples;
    unsigned int numberOfAttributes;
    string resultFile;
    bool usefulness;

    vector<string> t;
    vector<string> p;

    void openResultFile();
    void closeResultFile();

    long double computeProbabilityAttributeClass(const unsigned int& numberOfLevels,
                                               const unsigned int& attributeId,
                                               const string& classId);
    double computeProbabilityTrainingClass(const string& classId);

    void initializeT(const string& trueClass);
    void initializeP(const string& predictedClass);
    int getSizeT();
    int getSizeP();
    unsigned int minValue(const unsigned int& value1, const unsigned int& value2);
    int intersectionPT(const string& trueClass, const string& predictedClass);
    int numberOfLevels(const string& classId);

    long double applyClassifier(bool use_stdout);

    Classifier(const unsigned int& numberOfTrainingExamples,
              const unsigned int& numberOfTestExamples,
              const unsigned int& numberOfAttributes,
              const string& resultFile,
              const bool& usefulness)
        : numberOfTrainingExamples(numberOfTrainingExamples),
          numberOfTestExamples(numberOfTestExamples),
          numberOfAttributes(numberOfAttributes),
          resultFile(resultFile),
          usefulness(usefulness) {}

    ~Classifier() = default;
};

///! "middleware" used in c_nbayes.h for c conversion
long double nbayes(bool mlnp, bool usf, string trainingFile,
                string testFile, string resultFile);

#endif

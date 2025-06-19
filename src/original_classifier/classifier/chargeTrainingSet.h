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

#ifndef CHARGETRAININGSET_H
#define CHARGETRAININGSET_H

#include <vector>
#include <unordered_map>
#include <cmath>

#include "utils.h"

using namespace std;

class ChargeTrainingSet {
public:
    ifstream fin;
    string trainingFile;
    unsigned int numberOfAttributes;
    unsigned int numberOfTrainingExamples;
    unsigned int numberOfIndependentAttributeValues;
    unsigned int biggestLevel;
    bool mandatoryLeafNodePrediction;

    vector<int> attributeIndex;
    vector<int>::iterator it_attributeIndex;

    unordered_map<string, vector<unsigned int>> classFreq;
    unordered_map<string, vector<unsigned int>>::iterator it_classFreq;
    vector<unsigned int> counter;
    vector<unsigned int>::iterator it2_classFreq;
    vector<unsigned int>::iterator it_counter;

    unordered_map<string, double> classesForProbabilityEvaluation;
    unordered_map<string, double>::iterator it_classesForProbabilityEvaluation;

    vector<string> classPerLevel;
    vector<string>::iterator it_classPerLevel;

    void openTrainingFile();
    void closeTrainingFile();
    void getTrainingSet();

    void updateAttributeIndex(const unsigned int &index, const unsigned int &numberOfValues);
    void printAttributeIndex();
    int computeNumberOfIndependentAttributeValues();
    unsigned int getAttributeIndex(const unsigned int &attributeId);

    void initializeCounter();
    void updateCounter(const unsigned int &index);

    void updateClassFreq(const string &classId);
    unsigned int getAttributeValueClassFrequency(const unsigned int &attributeId, const unsigned int &attributeValue, const string &classId);
    unsigned int getClassFrequency(const string &classId);
    void printClassFreq();

    void setClassesForProbabilityEvaluation(const string &classId);
    bool notExistChild(const string &classId);
    void computeClassUsefulness();
    double getClassUsefulness(const string &classeId);
    void printClassesForProbabilityEvaluation();

    unsigned int initializeClassPerLevel(const string &classId);
    string getClassPerLevel(const int &index);
    string getSubsetClass(const int &level);
    void printClassPerLevel();

    ChargeTrainingSet(const string &trainingFile, const unsigned int &numberOfAttributes,
                     const unsigned int &numberOfTrainingExamples, const bool &mandatoryLeafNodePrediction) {
        this->trainingFile = trainingFile;
        this->numberOfAttributes = numberOfAttributes;
        this->numberOfTrainingExamples = numberOfTrainingExamples;
        this->mandatoryLeafNodePrediction = mandatoryLeafNodePrediction;

        attributeIndex.resize(numberOfAttributes + 1, 0);
    }

    ~ChargeTrainingSet() {}
};

#endif

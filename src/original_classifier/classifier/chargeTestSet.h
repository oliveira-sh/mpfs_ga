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

#ifndef CHARGETESTSET_H
#define CHARGETESTSET_H

#include <vector>

#include "utils.h"

using namespace std;

class ChargeTestSet {
public:
    ifstream fin;
    string testFile;
    unsigned int numberOfAttributes;
    unsigned int numberOfTestExamples;

    vector<vector<int>> testSet;
    vector<vector<int>>::iterator it_testSet;
    vector<int>::iterator it2_testSet;

    vector<string> classTestSet;
    vector<string>::iterator it_classTestSet;

    void openTestFile();
    void closeTestFile();

    void insertTestSet(const int &exampleId, const int &attributeId, const int &attributeValue);
    int getAttributeValueTestSet(const int &exampleId, const int &attributeId);
    void printTestSet();

    void insertClassTestSet(const int &exampleId, const string &attributeValue);
    string getClassTestSet(const int &exampleId);
    void printClassTestSet();

    void getTestSet();

    ChargeTestSet(const string &testFile, const int &numberOfTestExamples, const int &numberOfAttributes) {
        this->testFile = testFile;
        this->numberOfAttributes = numberOfAttributes;
        this->numberOfTestExamples = numberOfTestExamples;

        testSet.resize(numberOfTestExamples);
        for (it_testSet = testSet.begin(); it_testSet != testSet.end(); it_testSet++) {
            (*it_testSet).resize(numberOfAttributes - 1);
        }
        classTestSet.resize(numberOfTestExamples);
    }

    ~ChargeTestSet() {}
};

#endif

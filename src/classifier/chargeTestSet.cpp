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

#include <iostream>
#include "chargeTestSet.h"
#include "utils.h"

using namespace std;

void ChargeTestSet::openTestFile() {
    fin.open(testFile.c_str());
    while (!fin) {
        DEBUG_ERR("\nErro na abertura do arquivo " << testFile << " em chargeTestSet.cpp!!!" << endl);
        exit(1);
    }
}

void ChargeTestSet::closeTestFile() {
    fin.close();
}

void ChargeTestSet::insertTestSet(const int &exampleId, const int &attributeId, const int &attributeValue) {
    testSet[exampleId][attributeId] = attributeValue;
}

int ChargeTestSet::getAttributeValueTestSet(const int &exampleId, const int &attributeId) {
    return testSet[exampleId][attributeId];
}

void ChargeTestSet::printTestSet() {
    int exampleId = 0;
    for (it_testSet = testSet.begin(); it_testSet != testSet.end(); it_testSet++) {
        for (it2_testSet = (*it_testSet).begin(); it2_testSet != (*it_testSet).end(); it2_testSet++) {
            cout << *it2_testSet << " ";
        }
        cout << getClassTestSet(exampleId++) << endl;
    }
}

void ChargeTestSet::insertClassTestSet(const int &exampleId, const string &attributeValue) {
    classTestSet[exampleId] = attributeValue;
}

string ChargeTestSet::getClassTestSet(const int &exampleId) {
    return classTestSet[exampleId];
}

void ChargeTestSet::printClassTestSet() {
    for (it_classTestSet = classTestSet.begin(); it_classTestSet != classTestSet.end(); it_classTestSet++) {
        cout << *it_classTestSet << endl;
    }
}

void ChargeTestSet::getTestSet() {
    string line;
    int i, begin, end, exampleId = 0, attributeId = 0;

    openTestFile();

    while (lowercase(line.substr(0, 5)) != "@data") {
        getline(fin, line);
    }

    while (getline(fin, line)) {
        if (line.substr(0, 1) == "%" || (int)line.size() == 0) {
            continue;
        }

        i = -1;
        while (i < (int)line.size()) {
            begin = i;
            if (begin == line.rfind(",")) {
                end = line.size();
                insertClassTestSet(exampleId, line.substr(begin + 1, end - (begin + 1)));
            } else {
                end = line.find(",", begin + 1);
                insertTestSet(exampleId, attributeId, str2int(line.substr(begin + 1, end - (begin + 1))));
            }
            i = end;
            attributeId++;
        }
        exampleId++;
        attributeId = 0;
    }

    closeTestFile();
}

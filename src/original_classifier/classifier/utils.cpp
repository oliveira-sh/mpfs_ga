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

#include "utils.h"

string lowercase(string str) {
    for (char& c : str) {
        c = tolower(c);
    }
    return str;
}

int str2int(const string& value) {
    int result = 0;
    for (char c : value) {
        if (isdigit(c)) {
            result = result * 10 + (c - '0');
        }
    }
    return result;
}

void getDatasetsProfile(const string& trainingFile, const string& testFile,
                       unsigned int& numberOfTrainingExamples,
                       unsigned int& numberOfTestExamples,
                       unsigned int& numberOfAttributes) {

    numberOfTrainingExamples = 0;
    numberOfTestExamples = 0;
    numberOfAttributes = 1;
    unsigned int attributeCount = 0;

    string line, lastLine;
    ifstream fin(trainingFile);
    if (!fin.is_open()) {
        DEBUG_ERR("Error opening training file.");
    }

    while (getline(fin, line) && lowercase(line.substr(0, 5)) != "@data") {
        if (lowercase(line.substr(0, 10)) == "@attribute") {
            attributeCount++;
        }
    }

    while (getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        numberOfTrainingExamples++;
        lastLine = line;
    }
    fin.close();

    size_t pos = 0;
    while ((pos = lastLine.find(',', pos)) != string::npos) {
        numberOfAttributes++;
        pos++;
    }

    if (numberOfAttributes != attributeCount) {
        cout << numberOfAttributes << "\t" << attributeCount << endl;
        DEBUG_ERR("Training File Error: Inconsistent Number of Attributes.");
    }

    ifstream testFin(testFile);
    while (getline(testFin, line) && lowercase(line.substr(0, 5)) != "@data");
    while (getline(testFin, line)) {
        if (line.empty() || line[0] == '%') continue;
        numberOfTestExamples++;
    }
    testFin.close();
}

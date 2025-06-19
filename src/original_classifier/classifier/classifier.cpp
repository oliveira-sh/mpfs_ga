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

#include <algorithm>
#include <limits>

#include "classifier.h"

using namespace std;

ChargeTrainingSet *Classifier::auxCLCTR;
ChargeTestSet *Classifier::auxCLCTE;

void Classifier::openResultFile() {
    fout.open(resultFile.c_str());
    if (!fout) {
        cout << "\nError opening result file " << resultFile << endl;
        exit(1);
    }
}

void Classifier::closeResultFile() {
    fout.close();
}

long double Classifier::computeProbabilityAttributeClass(const unsigned int &exte,
                                                        const unsigned int &attributeId,
                                                        const string &classId) {
    unsigned int freq = auxCLCTR->getAttributeValueClassFrequency(attributeId,
                                                                  auxCLCTE->getAttributeValueTestSet(exte, attributeId),
                                                                  classId);
    unsigned int classFreq = auxCLCTR->getClassFrequency(classId);
    return static_cast<long double>(freq) / classFreq;
}

double Classifier::computeProbabilityTrainingClass(const string &classId) {
    return static_cast<double>(auxCLCTR->getClassFrequency(classId)) / numberOfTrainingExamples;
}

vector<string> Classifier::splitClass(const string &classStr) {
    vector<string> result;
    size_t start = 0, end = 0;

    while ((end = classStr.find('.', start)) != string::npos) {
        result.push_back(classStr.substr(start, end - start));
        start = end + 1;
    }
    result.push_back(classStr.substr(start));
    return result;
}

int Classifier::intersectionPT(const string &trueClass, const string &predictedClass) {
    vector<string> t = splitClass(trueClass);
    vector<string> p = splitClass(predictedClass);

    int count = 0;
    int minSize = min(t.size(), p.size());

    for (int i = 0; i < minSize && t[i] == p[i]; ++i) {
        ++count;
    }
    return count;
}

int Classifier::numberOfLevels(const string &classId) {
    return count(classId.begin(), classId.end(), '.') + 1;
}

long double Classifier::applyClassifier(bool use_stdout) {
    unsigned int numerator = 0, sumP = 0, sumT = 0, sumMinPT = 0;

    if (use_stdout) {
        openResultFile();
    }

    for (unsigned int exte = 0; exte < numberOfTestExamples; ++exte) {
        long double bestP = -numeric_limits<long double>::max();
        string bestClassId;

        for (const auto &classPair : auxCLCTR->classesForProbabilityEvaluation) {
            long double p = 0.0;

            for (unsigned int attributeId = 0; attributeId < numberOfAttributes - 1; ++attributeId) {
                long double temp = computeProbabilityAttributeClass(exte, attributeId, classPair.first);
                if (temp == 0.0) {
                    temp = 1.0 / numberOfTrainingExamples;
                }
                p += log10(temp);
            }

            p += log10(computeProbabilityTrainingClass(classPair.first));

            if (usefulness) {
                p = pow(10, p) * auxCLCTR->getClassUsefulness(classPair.first);
            } else {
                p = pow(10, p);
            }

            if (bestP < p) {
                bestP = p;
                bestClassId = classPair.first;
            }
        }

        string classOfTestExample = auxCLCTE->getClassTestSet(exte);
        int intersection = intersectionPT(classOfTestExample, bestClassId);

        numerator += intersection;

        vector<string> p_vec = splitClass(bestClassId);
        vector<string> t_vec = splitClass(classOfTestExample);

        unsigned int sizeP = p_vec.size();
        unsigned int sizeT = t_vec.size();

        sumP += sizeP;
        sumT += sizeT;
        sumMinPT += min(sizeP, sizeT);

        if (use_stdout) {
            fout << "Example " << exte << " (" << classOfTestExample << ") -> " << bestClassId << endl;
        }
    }

    double hP_new = static_cast<double>(numerator) / sumMinPT;
    double hP = static_cast<double>(numerator) / sumP;
    double hR = static_cast<double>(numerator) / sumT;
    long double result = 100 * (2 * hP * hR) / (hP + hR);

    if (use_stdout) {
        fout << "hP = " << hP_new * 100 << "\nhR = " << hR * 100
             << "\nhF = " << 100 * (2 * hP_new * hR) / (hP_new + hR) << endl;
        fout << "hP = " << hP * 100 << "\nhR = " << hR * 100
             << "\nhF = " << result;
        closeResultFile();
    }

    return result;
}

long double nbayes(bool mlnp, bool usf, string trainingFile, string testFile, string resultFile) {
    unsigned int numberOfAttributes, numberOfTrainingExamples, numberOfTestExamples;

    getDatasetsProfile(trainingFile, testFile, numberOfTrainingExamples,
                      numberOfTestExamples, numberOfAttributes);

    ChargeTrainingSet CTR(trainingFile, numberOfAttributes, numberOfTrainingExamples, mlnp);
    CTR.getTrainingSet();

    ChargeTestSet CTE(testFile, numberOfTestExamples, numberOfAttributes);
    CTE.getTestSet();

    bool use_stdout = !resultFile.empty();

    Classifier CL(numberOfTrainingExamples, numberOfTestExamples, numberOfAttributes, resultFile, usf);
    Classifier::auxCLCTR = &CTR;
    Classifier::auxCLCTE = &CTE;

    return CL.applyClassifier(use_stdout);
}

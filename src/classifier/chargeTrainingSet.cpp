#include <iostream>

#include "chargeTrainingSet.h"

using namespace std;

void ChargeTrainingSet::openTrainingFile() {
    fin.open(trainingFile);
    if (!fin.is_open()) {
        DEBUG_ERR("Error opening training file");
        exit(1);
    }
}

void ChargeTrainingSet::closeTrainingFile() {
    fin.close();
}

void ChargeTrainingSet::updateAttributeIndex(const unsigned int &index, const unsigned int &numberOfValues) {
    attributeIndex[index] = attributeIndex[index - 1] + numberOfValues;
}

void ChargeTrainingSet::printAttributeIndex() {
    unsigned int i = 0;
    for (it_attributeIndex = attributeIndex.begin(); it_attributeIndex != attributeIndex.end(); it_attributeIndex++) {
        cout << "Attribute " << i++ << "\t -> \t index " << *it_attributeIndex << endl;
    }
}

int ChargeTrainingSet::computeNumberOfIndependentAttributeValues() {
    return attributeIndex[numberOfAttributes - 1];
}

unsigned int ChargeTrainingSet::getAttributeIndex(const unsigned int &attributeId) {
    return attributeIndex[attributeId];
}

void ChargeTrainingSet::initializeCounter() {
    counter.clear();
    counter.resize(numberOfIndependentAttributeValues + 1, 0);
    counter[numberOfIndependentAttributeValues] = 1;
}

void ChargeTrainingSet::updateCounter(const unsigned int &index) {
    counter[index]++;
}

void ChargeTrainingSet::updateClassFreq(const string &classId) {
    if ((it_classFreq = classFreq.find(classId)) == classFreq.end()) {
        classFreq[classId] = counter;
    } else {
        it_counter = counter.begin();
        for (it2_classFreq = it_classFreq->second.begin(); it2_classFreq != it_classFreq->second.end(); it2_classFreq++) {
            if ((*it_counter) == 1) (*it2_classFreq)++;
            it_counter++;
        }
    }
}

unsigned int ChargeTrainingSet::getAttributeValueClassFrequency(const unsigned int &attributeId, const unsigned int &attributeValue, const string &classId) {
    if (classFreq.find(classId) != classFreq.end()) {
        return classFreq[classId][getAttributeIndex(attributeId) + attributeValue];
    }
    return 0;
}

unsigned int ChargeTrainingSet::getClassFrequency(const string &classId) {
    if (classFreq.find(classId) != classFreq.end()) {
        return classFreq[classId][numberOfIndependentAttributeValues];
    }
    return 0;
}

void ChargeTrainingSet::printClassFreq() {
    int index;
    for (it_classFreq = classFreq.begin(); it_classFreq != classFreq.end(); it_classFreq++) {
        cout << "Class " << it_classFreq->first << endl;
        index = 0;
        for (it2_classFreq = it_classFreq->second.begin(); it2_classFreq != it_classFreq->second.end(); it2_classFreq++) {
            cout << index << "\t" << *it2_classFreq << endl;
            index++;
        }
    }
}

void ChargeTrainingSet::setClassesForProbabilityEvaluation(const string &classId) {
    int begin = -1, end = -1;
    string fatherNode, subsetClassId;

    if (classesForProbabilityEvaluation.find(classId) == classesForProbabilityEvaluation.end()) {
        if (mandatoryLeafNodePrediction) {
            if (notExistChild(classId)) {
                classesForProbabilityEvaluation[classId] = 1;
                fatherNode.clear();
                while ((end = classId.find(".", begin + 1)) != string::npos) {
                    fatherNode += classId.substr(begin + 1, end - begin - 1);
                    classesForProbabilityEvaluation.erase(fatherNode);
                    begin = end;
                    fatherNode += ".";
                }
            }
        } else {
            subsetClassId.clear();
            while ((end = classId.find(".", begin + 1)) != string::npos) {
                subsetClassId += classId.substr(begin + 1, end - begin - 1);
                classesForProbabilityEvaluation[subsetClassId] = 1;
                begin = end;
                subsetClassId += ".";
            }
            classesForProbabilityEvaluation[classId] = 1;
        }
    }
}

bool ChargeTrainingSet::notExistChild(const string &classId) {
    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        if (it_classesForProbabilityEvaluation->first.size() > classId.size() &&
            it_classesForProbabilityEvaluation->first.substr(0, (int)classId.size()) == classId) {
            return false;
        }
    }
    return true;
}

void ChargeTrainingSet::computeClassUsefulness() {
    int begin = -1, end = -1, maxTreeSize = 1;
    string fatherNode, classeId;

    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        it_classesForProbabilityEvaluation->second = 1;
    }

    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        fatherNode.clear();
        begin = -1;
        end = -1;
        classeId = it_classesForProbabilityEvaluation->first;
        while ((end = classeId.find(".", begin + 1)) != string::npos) {
            fatherNode += classeId.substr(begin + 1, end - begin - 1);
            if (classesForProbabilityEvaluation.find(fatherNode) != classesForProbabilityEvaluation.end()) {
                classesForProbabilityEvaluation[fatherNode]++;
            }
            begin = end;
            fatherNode += ".";
        }
    }

    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        if (it_classesForProbabilityEvaluation->second > maxTreeSize) {
            maxTreeSize = it_classesForProbabilityEvaluation->second;
        }
    }

    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        classesForProbabilityEvaluation[it_classesForProbabilityEvaluation->first] =
            1 - log2(it_classesForProbabilityEvaluation->second) / log2(maxTreeSize + 1);
    }
}

double ChargeTrainingSet::getClassUsefulness(const string &classeId) {
    return classesForProbabilityEvaluation[classeId];
}

void ChargeTrainingSet::printClassesForProbabilityEvaluation() {
    DEBUG_DBG("Training Classes");
    for (it_classesForProbabilityEvaluation = classesForProbabilityEvaluation.begin();
         it_classesForProbabilityEvaluation != classesForProbabilityEvaluation.end();
         it_classesForProbabilityEvaluation++) {
        cout << it_classesForProbabilityEvaluation->first << "\t" << it_classesForProbabilityEvaluation->second << endl;
    }
}

unsigned int ChargeTrainingSet::initializeClassPerLevel(const string &classId) {
    int begin = -1, end = -1;

    classPerLevel.clear();
    while ((end = classId.find(".", begin + 1)) != string::npos) {
        classPerLevel.push_back(classId.substr(begin + 1, end - begin - 1));
        begin = end;
    }
    classPerLevel.push_back(classId.substr(begin + 1, classId.size() - begin - 1));
    return classPerLevel.size();
}

string ChargeTrainingSet::getClassPerLevel(const int &index) {
    return classPerLevel[index];
}

string ChargeTrainingSet::getSubsetClass(const int &level) {
    string subsetClassId;
    subsetClassId.clear();
    for (int i = 0; i < level; i++) {
        subsetClassId += getClassPerLevel(i);
        subsetClassId += ".";
    }
    return subsetClassId.substr(0, (int)subsetClassId.size() - 1);
}

void ChargeTrainingSet::printClassPerLevel() {
    DEBUG_DBG("Class separated per level: ");
    for (it_classPerLevel = classPerLevel.begin(); it_classPerLevel != classPerLevel.end(); it_classPerLevel++) {
        cout << *it_classPerLevel << "\t";
    }
    cout << endl;
}

void ChargeTrainingSet::getTrainingSet() {
    int begin = -1, end = -1, stop = 0;
    unsigned int numberOfValues = 0, attributeId = 0;
    unsigned int numberOfLevels = 0;
    string line, lastLine, classId, attributeValue, subsetClassId;
    numberOfIndependentAttributeValues = 0;
    biggestLevel = 0;

    openTrainingFile();

    while (getline(fin, line) && lowercase(line.substr(0, 5)) != "@data") {
        if (lowercase(line.substr(0, 10)) != "@attribute") {
            continue;
        }
        numberOfValues = 1;
        begin = line.find("{", 9);
        while ((end = line.find(",", begin + 1)) != string::npos) {
            numberOfValues++;
            begin = end;
        }
        updateAttributeIndex(attributeId + 1, numberOfValues);
        attributeId++;
        lastLine = line;
    }

    numberOfIndependentAttributeValues = computeNumberOfIndependentAttributeValues();

    begin = lastLine.find("{", 9);
    while ((end = lastLine.find(",", begin + 1)) != string::npos) {
        setClassesForProbabilityEvaluation(lastLine.substr(begin + 1, end - begin - 1));
        begin = end;
    }
    end = lastLine.find("}", begin + 1);
    setClassesForProbabilityEvaluation(lastLine.substr(begin + 1, end - begin - 1));

    while (getline(fin, line)) {
        if (line.substr(0, 1) == "%" || (int)line.size() == 0) {
            continue;
        }
        initializeCounter();

        begin = line.rfind(",");
        end = line.size();
        classId = line.substr(begin + 1, end - begin - 1);

        stop = line.rfind(",");
        begin = -1;
        attributeId = 0;
        while (begin < stop) {
            end = line.find(",", begin + 1);
            attributeValue = line.substr(begin + 1, end - (begin + 1));
            if (str2int(attributeValue) <= (getAttributeIndex(attributeId + 1) - getAttributeIndex(attributeId))) {
                updateCounter(getAttributeIndex(attributeId) + str2int(attributeValue));
            } else {
                string err = "Training File Error: Inconsistent Attribute Value for AttributeId: " + to_string(attributeId) + "\n";
                DEBUG_ERR(err);
                exit(1);
            }
            begin = end;
            attributeId++;
        }

        numberOfLevels = initializeClassPerLevel(classId);
        if (numberOfLevels > biggestLevel) biggestLevel = numberOfLevels;

        for (unsigned int i = 1; i <= numberOfLevels; i++) {
            updateClassFreq(getSubsetClass(i));
        }
    }

    computeClassUsefulness();
    closeTrainingFile();
}

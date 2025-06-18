#include "classifier.h"
#include <string.h>
#include <string>

using namespace std;

ChargeTrainingSet *Classifier::auxCLCTR;
ChargeTestSet *Classifier::auxCLCTE;

void Classifier::openResultFile(){
  fout.open(resultFile.c_str());
  if(!fout){
	cout << "\nError opening result file " << resultFile << endl;
	exit(1);
  }
}

void Classifier::closeResultFile(){
  fout.close();
}

//#################Compute Probability Section########################
long double Classifier::computeProbabilityAttributeClass(const unsigned int &exte, const unsigned int &attributeId, const string &classId){
  long double prod = 0;

  prod = (double)auxCLCTR->getAttributeValueClassFrequency(attributeId,auxCLCTE->getAttributeValueTestSet(exte,attributeId),classId)/(double)auxCLCTR->getClassFrequency(classId);

  return(prod);
}

double Classifier::computeProbabilityTrainingClass(const string &classId){
  return((double)auxCLCTR->getClassFrequency(classId)/(double)numberOfTrainingExamples);
}
//######################Performance Measures##########################
void Classifier::initializeT(const string &trueClass){
  int begin = -1, end = -1;
  t.clear();
  while((end=trueClass.find(".",begin+1)) != string::npos){
	t.push_back(trueClass.substr(begin+1,end-begin-1));
	begin = end;
  }
  t.push_back(trueClass.substr(begin+1,trueClass.size()-begin-1));
}

void Classifier::initializeP(const string &predictedClass){
  int begin = -1, end = -1;
  p.clear();
  while((end=predictedClass.find(".",begin+1)) != string::npos){
	p.push_back(predictedClass.substr(begin+1,end-begin-1));
	begin = end;
  }
  p.push_back(predictedClass.substr(begin+1,predictedClass.size()-begin-1));
}

int Classifier::getSizeT(){
  return((int)t.size());
}

int Classifier::getSizeP(){
  return((int)p.size());
}

unsigned int Classifier::minValue(const unsigned int &value1, const unsigned int &value2){
	if(value1 < value2) return(value1);
	else return(value2);
}

int Classifier::intersectionPT(const string &trueClass, const string &predictedClass){
  int count = 0, small = 0, sizeP = 0, sizeT = 0;

  initializeT(trueClass);
  initializeP(predictedClass);
  //Comparando as classes verdadeira e predita
  sizeT = getSizeT();
  sizeP = getSizeP();
  if(sizeP < sizeT) small = sizeP;
  else small = sizeT;

  for(int i = 0; i < small; i++){
	if(t[i] == p[i]) count++;
	else break;
  }
  return(count);
}

//#########################Support Section###############################
int Classifier::numberOfLevels(const string &classId){
  int begin = -1, count = 1;
  size_t found;
  while((found = classId.find(".",begin+1)) != string::npos){
	count++;
	begin = (int)found;
  }
  return(count);
}

//#########################Main Section###############################
long double Classifier::applyClassifier(bool use_stdout){
    unsigned int exte = 0,
    attributeId = 0,
    numerator = 0,
    sumP = 0,
    sumT = 0,
    sumMinPT = 0,
    sizeP,
    sizeT;

    string bestClassId, classOfTestExample;
    long double p, bestP, temp, hP, hR, hP_new, result;

    if (use_stdout) {
        openResultFile();
    }

    for(exte = 0; exte < numberOfTestExamples; exte++){//Para cada instancia da base de teste...
        bestP = -numeric_limits<long double>::max();
    		bestClassId.clear();
    		for(auxCLCTR->it_classesForProbabilityEvaluation = auxCLCTR->classesForProbabilityEvaluation.begin(); auxCLCTR->it_classesForProbabilityEvaluation != auxCLCTR->classesForProbabilityEvaluation.end(); auxCLCTR->it_classesForProbabilityEvaluation++){//Para cada uma das classes em trainingSetClasses...
    			p = 0;
    			for(attributeId = 0; attributeId < numberOfAttributes-1; attributeId++){
    				temp = computeProbabilityAttributeClass(exte,attributeId,auxCLCTR->it_classesForProbabilityEvaluation->first);
    				if(temp == 0) temp = 1.0/(double)numberOfTrainingExamples;
    				p+= log10(temp);
    			}

    			//Calculo incluindo a probabilidade da classe
    			p+= log10(computeProbabilityTrainingClass(auxCLCTR->it_classesForProbabilityEvaluation->first));

    			//Calculo incluindo o Usefulness
    			if(usefulness)
    				p = pow(10,p) * auxCLCTR->getClassUsefulness(auxCLCTR->it_classesForProbabilityEvaluation->first);
    			else
    				p = pow(10,p);

    			if(bestP < p){
    			bestP = p;
    			bestClassId = auxCLCTR->it_classesForProbabilityEvaluation->first;
    			}
    		}
    		classOfTestExample = auxCLCTE->getClassTestSet(exte);
    		numerator+=intersectionPT(classOfTestExample,bestClassId); //tirei a modificacao para R.
    		sizeP = getSizeP();
    		sizeT = getSizeT();
            sumP+= sizeP;
            sumT+= sizeT;
    		sumMinPT+= minValue(sizeP,sizeT);

            if(use_stdout) {
                fout << "Example " << exte << " ("<< auxCLCTE->getClassTestSet(exte)  << ") -> " << bestClassId << endl;
            }


    }
    hP_new = (double)numerator/(double)sumMinPT;
    hP = (double)numerator/(double)sumP;
    hR = (double)numerator/(double)sumT;
    result = 100*(2*hP*hR)/(hP+hR);

    if (use_stdout) {
        fout << "hP = " << hP_new*100 << "\nhR = " << hR*100 << "\nhF = " << 100*(2*hP_new*hR)/(hP_new+hR) << endl;
        fout << "hP = " << hP*100 << "\nhR = " << hR*100 << "\nhF = " << result;
        closeResultFile();
    }

    return result;
}

long double nbayes(bool mlnp, bool usf, string trainingFile, string testFile, string resultFile){
    unsigned int numberOfAttributes;
    unsigned int numberOfTrainingExamples;
    unsigned int numberOfTestExamples;
    long double result;

    	getDatasetsProfile(trainingFile,
         testFile,
         numberOfTrainingExamples,
         numberOfTestExamples,
         numberOfAttributes);

    	ChargeTrainingSet *CTR;
    	CTR = new ChargeTrainingSet(trainingFile,numberOfAttributes,numberOfTrainingExamples, mlnp);
    	CTR->getTrainingSet();

    	ChargeTestSet *CTE;
    	CTE = new ChargeTestSet(testFile,numberOfTestExamples,numberOfAttributes);
    	CTE->getTestSet();

        bool use_stdout = resultFile != "";

    	Classifier *CL;
    	CL = new Classifier(numberOfTrainingExamples, numberOfTestExamples, numberOfAttributes, resultFile, usf);
    	Classifier::auxCLCTR = CTR;
    	Classifier::auxCLCTE = CTE;
    	result = CL->applyClassifier(use_stdout);

    	delete CL;
    	delete CTE;
    	delete CTR;
    	return result;
}

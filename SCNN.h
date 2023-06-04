#pragma once
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#define e 2.71828
//#define DEBUG_READ_DATA_ALL
using namespace std;

template<typename T_1>
class SCNN
{
protected:
	const int ALRConst = 10;
	const int ALRConstAv = 30;
	const float relluKoeficient = 0.01;
	const float Beta = 0.9;
	const float I_Beta = 0.1;
	const float de = 0.0000000001;
	T_1 Alfa;


	bool flag_Momentum;
	bool flag_RMS;
	bool flag_learning;
	bool flag_learningActivate;
	bool flag_alstart;

	int NNIterator;
	int backPropIterator;
	int iterationReadData;
	int iteratorDirectPass;
	int iteratorALR;
	int iteratorALRRemember;
	int numberEpoch;
	int Epoch;
	int pixelsPerImage;
	int miniBatch;
	int layers;
	int mooving;

	T_1 examplsNumber;
	T_1 lamda;
	T_1 sumE;
	T_1 CE;
	T_1 CEPerEpoch;
	T_1 CEPerEpochTMP;

	T_1 CEAverage;
	T_1 CEDifferenceAv;
	T_1 CEDivider;
	T_1 CEAvLast;
	queue<T_1> CEQueue;
	queue<T_1> CEDifferenceLast;

	vector<T_1> dError;
	vector<T_1> dErrorAverage;
	vector<int> sizePerLayers;
	vector<T_1> dataSet;
	vector<T_1> originalData;
	vector<vector<T_1>> dataSetAll;
	vector<vector<T_1>> neurons;
	vector<vector<T_1>> dEdS;
	vector<vector<T_1>> sumNeurons;
	vector<vector<T_1>> bias;
	vector<vector<T_1>> biasGradientAverage;
	vector<vector<vector<T_1>>> weights;
	vector<vector<vector<T_1>>> weightsGradientAverage;

	vector<vector<T_1>> RMSBias;
	vector<vector<T_1>> momentumBias;
	vector<vector<vector<T_1>>> RMSWeights;
	vector<vector<vector<T_1>>> momentumWeights;

	inline T_1 NormalDistribution(int l);
	inline T_1 ReLU(T_1 x) noexcept;
	inline T_1 dReLU(T_1 x) noexcept;
	inline void setVectorMomentum() noexcept;
	inline void setVectorRMS() noexcept;
	inline void SoftMax();
	inline T_1 CroosEntropy(const vector<T_1>& resultData);
	inline void functiondEdS();
	inline void functionNMdEdS();
	void errorOutput();
public:
	enum SCNNMethod { SGDMethod, MOMENTUM, NESTEROVMOMENTUM, RMSPROP, ADAM, ADAMNM};
	SCNN();
	SCNN(int layers, vector<int>& sizePerLayers);
	SCNN(const SCNN& obj_0);
	~SCNN();


	const SCNN& operator=(const SCNN& obj_0);
	void directPass();
	void SqrError();
	void adaptiveLR();
	void SGD();
	void Momentum();
	void NesterovMomentum();
	void RMSprop();
	void Adam();
	void AdamNM();
	void learning(int epoch, enum SCNNMethod);
	void learning(int epoch);

	void readDataAll(string fileName = "lib_MNIST_binary.txt");
	void readData(string fileName = "lib_MNIST_binary.txt");
	void fileNameFunction(string name);
	void setPixelPerImage(int pixelPerImage);
	void setMiniBatch(int mini_batch);
};

template<typename T_1>
SCNN<T_1>::SCNN() {
	srand(time(NULL));
	flag_Momentum = false;
	flag_RMS = false;
	flag_learning = false;
	flag_alstart = false;
	flag_learningActivate = false;
	lamda = 0.003;
	CEDivider = 2;
	pixelsPerImage = 784;
	miniBatch = 128;
	Epoch = 0;
	NNIterator = 0;
	iteratorDirectPass = 0;
	backPropIterator = 0;
	sumE = 0;
	CE = 0;
	CEPerEpoch = 0;
	CEPerEpochTMP = 0;
	CEAverage = 0;
	CEDifferenceAv = 0;
	CEAvLast = 0;
	examplsNumber = 0;
	mooving = 0;
	iterationReadData = 0;
	iteratorALR = 0;
	iteratorALRRemember = 0;
	numberEpoch = 0;
	Alfa = 1;
}

template<typename T_1>
SCNN<T_1>::SCNN(int layers, vector<int>& sizePerLayers) {
	srand(time(NULL));
	flag_Momentum = false;
	flag_RMS = false;
	flag_learning = false;
	flag_alstart = false;
	flag_learningActivate = false;
	lamda = 0.003;
	CEDivider = 2;
	pixelsPerImage = 784;
	miniBatch = 128;
	Epoch = 0;
	NNIterator = 0;
	iteratorDirectPass = 0;
	backPropIterator = 0;
	sumE = 0;
	CE = 0;
	CEPerEpoch = 0;
	CEPerEpochTMP = 0;
	CEAverage = 0;
	CEDifferenceAv = 0;
	CEAvLast = 0;
	examplsNumber = 0;
	mooving = 0;
	iterationReadData = 0;
	iteratorALR = 0;
	iteratorALRRemember = 0;
	numberEpoch = 0;
	Alfa = 1;

	this->layers = layers;
	this->sizePerLayers = sizePerLayers;

	CEQueue = queue<T_1>();
	CEDifferenceLast = queue<T_1>();
	originalData = vector<T_1>(sizePerLayers[layers - 1], 0);
	dError = vector<T_1>(sizePerLayers[layers - 1], 0);
	dErrorAverage = vector<T_1>(sizePerLayers[layers - 1], 0);

	neurons = vector<vector<T_1>>(layers);
	sumNeurons = vector<vector<T_1>>(layers - 1);
	dEdS = vector<vector<T_1>>(layers - 1);

	bias = vector<vector<T_1>>(layers - 1);
	biasGradientAverage = vector<vector<T_1>>(layers - 1);

	weights = vector<vector<vector<T_1>>>(layers - 1);
	weightsGradientAverage = vector<vector<vector<T_1>>>(layers - 1);

	for (int l = 0; l < layers; ++l) {
		neurons[l] = vector<T_1>(sizePerLayers[l], 0);
	}

	for (int l = 0; l < layers - 1; ++l) {
		sumNeurons[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		dEdS[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		bias[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		biasGradientAverage[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		weights[l] = vector<vector<T_1>>(sizePerLayers[l]);
		weightsGradientAverage[l] = vector<vector<T_1>>(sizePerLayers[l]);
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			weights[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
			weightsGradientAverage[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
		}
	}

	for (int l = 0; l < layers - 1; ++l) {
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
				weights[l][i][j] = NormalDistribution(l);
			}
		}
	}
}

template<typename T_1>
SCNN<T_1>::SCNN(const SCNN& obj_0) {
	flag_Momentum = obj_0.flag_Momentum;
	flag_RMS = obj_0.flag_RMS;
	flag_learning = obj_0.flag_learning;
	flag_alstart = obj_0.flag_alstart;
	flag_learningActivate = obj_0.flag_learningActivate;
	lamda = obj_0.lamda;;
	CEDivider = obj_0.CEDivider;
	pixelsPerImage = obj_0.pixelsPerImage;
	miniBatch = obj_0.miniBatch;
	Epoch = obj_0.Epoch;
	NNIterator = obj_0.NNIterator;
	iteratorDirectPass = obj_0.iteratorDirectPass;
	backPropIterator = obj_0.backPropIterator;
	sumE = obj_0.sumE;
	CE = obj_0.CE;
	CEPerEpoch = obj_0.CEPerEpoch;
	CEPerEpochTMP = obj_0.CEPerEpochTMP;
	CEAverage = obj_0.CEAverage;
	CEDifferenceAv = obj_0.CEDifferenceAv;
	CEAvLast = obj_0.CEAvLast;
	examplsNumber = obj_0.examplsNumber;
	mooving = obj_0.mooving;
	iterationReadData = obj_0.iterationReadData;
	iteratorALR = obj_0.iteratorALR;
	iteratorALRRemember = obj_0.iteratorALRRemember;
	numberEpoch = obj_0.numberEpoch;
	Alfa = 1;

	this->layers = obj_0.layers;
	this->sizePerLayers = obj_0.sizePerLayers;

	CEQueue = queue<T_1>();
	CEDifferenceLast = queue<T_1>();
	originalData = vector<T_1>(sizePerLayers[layers - 1], 0);
	dError = vector<T_1>(sizePerLayers[layers - 1], 0);
	dErrorAverage = vector<T_1>(sizePerLayers[layers - 1], 0);

	neurons = vector<vector<T_1>>(layers);
	sumNeurons = vector<vector<T_1>>(layers - 1);
	dEdS = vector<vector<T_1>>(layers - 1);

	bias = vector<vector<T_1>>(layers - 1);
	biasGradientAverage = vector<vector<T_1>>(layers - 1);

	weights = vector<vector<vector<T_1>>>(layers - 1);
	weightsGradientAverage = vector<vector<vector<T_1>>>(layers - 1);

	for (int l = 0; l < layers; ++l) {
		neurons[l] = vector<T_1>(sizePerLayers[l], 0);
	}

	for (int l = 0; l < layers - 1; ++l) {
		sumNeurons[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		dEdS[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		bias[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		biasGradientAverage[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		weights[l] = vector<vector<T_1>>(sizePerLayers[l]);
		weightsGradientAverage[l] = vector<vector<T_1>>(sizePerLayers[l]);
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			weights[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
			weightsGradientAverage[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
		}
	}

	for (int l = 0; l < layers - 1; ++l) {
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
				weights[l][i][j] = NormalDistribution(l);
			}
		}
	}
	if (obj_0.layers > 0) {
		neurons = obj_0.neurons;
		weights = obj_0.weights;
		bias = obj_0.bias;
		dEdS = obj_0.dEdS;
		biasGradientAverage = obj_0.biasGradientAverage;
		weightsGradientAverage = obj_0.weightsGradientAverage;
		sumNeurons = obj_0.sumNeurons;
		CEQueue = obj_0.CEQueue;
		CEDifferenceLast = obj_0.CEDifferenceLast;
		originalData = obj_0.originalData;
		dataSetAll = obj_0.dataSetAll;
		dataSet = obj_0.dataSet;
	}
}

template<typename T_1>
SCNN<T_1>::~SCNN() {

}

template<typename T_1>
const SCNN<T_1>& SCNN<T_1>::operator=(const SCNN& obj_0) {
	flag_Momentum = obj_0.flag_Momentum;
	flag_RMS = obj_0.flag_RMS;
	flag_learning = obj_0.flag_learning;
	flag_alstart = obj_0.flag_alstart;
	flag_learningActivate = obj_0.flag_learningActivate;
	lamda = obj_0.lamda;;
	CEDivider = obj_0.CEDivider;
	pixelsPerImage = obj_0.pixelsPerImage;
	miniBatch = obj_0.miniBatch;
	Epoch = obj_0.Epoch;
	NNIterator = obj_0.NNIterator;
	iteratorDirectPass = obj_0.iteratorDirectPass;
	backPropIterator = obj_0.backPropIterator;
	sumE = obj_0.sumE;
	CE = obj_0.CE;
	CEPerEpoch = obj_0.CEPerEpoch;
	CEPerEpochTMP = obj_0.CEPerEpochTMP;
	CEAverage = obj_0.CEAverage;
	CEDifferenceAv = obj_0.CEDifferenceAv;
	CEAvLast = obj_0.CEAvLast;
	examplsNumber = obj_0.examplsNumber;
	mooving = obj_0.mooving;
	iterationReadData = obj_0.iterationReadData;
	iteratorALR = obj_0.iteratorALR;
	iteratorALRRemember = obj_0.iteratorALRRemember;
	numberEpoch = obj_0.numberEpoch;
	Alfa = 1;

	this->layers = obj_0.layers;
	this->sizePerLayers = obj_0.sizePerLayers;

	CEQueue = queue<T_1>();
	CEDifferenceLast = queue<T_1>();
	originalData = vector<T_1>(sizePerLayers[layers - 1], 0);
	dError = vector<T_1>(sizePerLayers[layers - 1], 0);
	dErrorAverage = vector<T_1>(sizePerLayers[layers - 1], 0);

	neurons = vector<vector<T_1>>(layers);
	sumNeurons = vector<vector<T_1>>(layers - 1);
	dEdS = vector<vector<T_1>>(layers - 1);

	bias = vector<vector<T_1>>(layers - 1);
	biasGradientAverage = vector<vector<T_1>>(layers - 1);

	weights = vector<vector<vector<T_1>>>(layers - 1);
	weightsGradientAverage = vector<vector<vector<T_1>>>(layers - 1);

	for (int l = 0; l < layers; ++l) {
		neurons[l] = vector<T_1>(sizePerLayers[l], 0);
	}

	for (int l = 0; l < layers - 1; ++l) {
		sumNeurons[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		dEdS[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		bias[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		biasGradientAverage[l] = vector<T_1>(sizePerLayers[l + 1], 0);
		weights[l] = vector<vector<T_1>>(sizePerLayers[l]);
		weightsGradientAverage[l] = vector<vector<T_1>>(sizePerLayers[l]);
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			weights[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
			weightsGradientAverage[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
		}
	}

	for (int l = 0; l < layers - 1; ++l) {
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
				weights[l][i][j] = NormalDistribution(l);
			}
		}
	}
	if (obj_0.layers > 0) {
		neurons = obj_0.neurons;
		weights = obj_0.weights;
		bias = obj_0.bias;
		dEdS = obj_0.dEdS;
		biasGradientAverage = obj_0.biasGradientAverage;
		weightsGradientAverage = obj_0.weightsGradientAverage;
		sumNeurons = obj_0.sumNeurons;
		CEQueue = obj_0.CEQueue;
		CEDifferenceLast = obj_0.CEDifferenceLast;
		originalData = obj_0.originalData;
		dataSetAll = obj_0.dataSetAll;
		dataSet = obj_0.dataSet;
	}
	return *this;
}

template<typename T_1>
void SCNN<T_1>::setPixelPerImage(int pixelPerImage) {
	this->pixelsPerImage = pixelPerImage;
	sizePerLayers[0] = pixelPerImage;
}

template<typename T_1>
void SCNN<T_1>::setMiniBatch(int mini_batch) {
	miniBatch = mini_batch;
}

template<typename T_1>
void SCNN<T_1>::setVectorMomentum() noexcept {
	if (layers > 0) {
		momentumWeights = vector<vector<vector<T_1>>>(layers - 1);
		momentumBias = vector<vector<T_1>>(layers - 1);

		for (int l = 0; l < layers - 1; ++l) {
			momentumWeights[l] = vector<vector<T_1>>(sizePerLayers[l]);
			momentumBias[l] = vector<T_1>(sizePerLayers[l + 1], 0);
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				momentumWeights[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
			}
		}
		flag_Momentum = true;
	}
}

template<typename T_1>
void SCNN<T_1>::setVectorRMS() noexcept {
	if (layers > 0) {
		RMSWeights = vector<vector<vector<T_1>>>(layers - 1);
		RMSBias = vector<vector<T_1>>(layers - 1);

		for (int l = 0; l < layers - 1; ++l) {
			RMSWeights[l] = vector<vector<T_1>>(sizePerLayers[l]);
			RMSBias[l] = vector<T_1>(sizePerLayers[l + 1], 0);
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				RMSWeights[l][i] = vector<T_1>(sizePerLayers[l + 1], 0);
			}
		}
		flag_RMS = true;
	}
}

template<typename T_1>
T_1 SCNN<T_1>::NormalDistribution(int l) {
	T_1 x = 0;
	T_1 y = 0;
	T_1 s = 1.1;
	while (s < 0 || s > 1) {
		x = T_1(rand() % 2000 - 1000) / 1000;
		y = T_1(rand() % 2000 - 1000) / 1000;
		s = x * x + y * y;
	}
	x = x * sqrt(-2.0 * log(s) / s);
	y = y * sqrt(-2.0 * log(s) / s);
	return (x + y) * sqrt(2.0 / sizePerLayers[l]) / 2;
}

template<typename T_1>
T_1 SCNN<T_1>::ReLU(T_1 x) noexcept {
	if (x < 0) return x * relluKoeficient;
	//if (x > 1) return 1 + x * relluKoeficient;
	return x;
}

template<typename T_1>
T_1 SCNN<T_1>::dReLU(T_1 x)  noexcept {
	if (x < 0) return -relluKoeficient;
	//if (x > 1) return relluKoeficient;
	return 1;
}

template<typename T_1>
void SCNN<T_1>::SoftMax() {
	sumE = 0;
	for (int i = 0; i < neurons[layers - 1].size(); ++i) {
		neurons[layers - 1][i] = pow(e, sumNeurons[layers - 2][i]);
		sumE += neurons[layers - 1][i];
	}
	for (int i = 0; i < neurons[layers - 1].size(); ++i) {
		neurons[layers - 1][i] = neurons[layers - 1][i] / sumE;
	}
}

template<typename T_1>
void SCNN<T_1>::errorOutput() {
	
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) {
		dErrorAverage[i] /= backPropIterator;
		dErrorAverage[i] = sqrt(dErrorAverage[i]);
	}
	cout << "\nITERATION: " << NNIterator << endl;
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) {
		cout << "dError[" << i << "]: " << dError[i] << "\t";
		if (i % 3 == 0) cout << endl;
	}
	cout << endl;
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) {
		cout << "dErrorAverage[" << i << "]: " << dErrorAverage[i] << "\t";
		if (i % 3 == 0) cout << endl;
	}
	cout << "Cross Entropy: " << CE << "\tCross Entropy per EPOCH[" << Epoch - 1 << "]: " << CEPerEpoch << endl;
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) dErrorAverage[i] = 0;
	if (Epoch >= 10) system("pause");
}

template<typename T_1>
T_1 SCNN<T_1>::CroosEntropy(const vector<T_1>& originalData) {
	T_1 sum = 0;
	for (int i = 0; i < neurons[layers - 1].size(); ++i) {
		sum += originalData[i] * sumNeurons[layers - 2][i];
	}
	return -sum + log(sumE);
}

template<typename T_1>
void SCNN<T_1>::directPass() {
	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				sumNeurons[l][j] += neurons[l][i] * weights[l][i][j];
			}
			sumNeurons[l][j] += bias[l][j];
			if (l + 1 < layers - 1) neurons[l + 1][j] = ReLU(sumNeurons[l][j]);
		}
	}

	SoftMax();
	CE += CroosEntropy(originalData);
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) {
		dError[i] = neurons[layers - 1][i] - originalData[i];
		dErrorAverage[i] += pow(dError[i], 2);
	}
	//++iteratorDirectPass;
}

template<typename T_1>
void SCNN<T_1>::SqrError() {
	T_1 error_tmp = 0;
	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				sumNeurons[l][j] += neurons[l][i] * weights[l][i][j];
			}
			sumNeurons[l][j] += bias[l][j];
			neurons[l + 1][j] = ReLU(sumNeurons[l][j]);
		}
	}
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) {
		dError[i] = pow(neurons[layers - 1][i] - originalData[i], 2);
		dErrorAverage[i] += dError[i];
		error_tmp += dError[i];
	}
	CE += error_tmp / (layers - 1);
}

template<typename T_1>
void SCNN<T_1>::functiondEdS() {
	++backPropIterator;
	for (int j = 0; j < sizePerLayers[layers - 1]; ++j)  dEdS[layers - 2][j] = dError[j];
			
	for (int l = layers - 2; l > 0; --l) {
		if (l - 1 < 0) break;
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				dEdS[l - 1][i] += weights[l][i][j] * dEdS[l][j];
			}
		}
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			dEdS[l - 1][i] *= dReLU(sumNeurons[l - 1][i]);
		}
	}


	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			biasGradientAverage[l][j] += dEdS[l][j];
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				weightsGradientAverage[l][i][j] += neurons[l][i] * dEdS[l][j];
			}
			dEdS[l][j] = 0;
		}
	}

	for (int l = layers - 2; l >= 0; --l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			sumNeurons[l][j] = 0;
		}
	}

}

template<typename T_1>
void SCNN<T_1>::functionNMdEdS() {
	++backPropIterator;
	for (int j = 0; j < sizePerLayers[layers - 1]; ++j)  dEdS[layers - 2][j] = dError[j];

	for (int l = layers - 2; l > 0; --l) {
		if (l - 1 < 0) break;
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				dEdS[l - 1][i] += (weights[l][i][j] - lamda * momentumWeights[l][i][j]) * dEdS[l][j];
			}
		}
		for (int i = 0; i < sizePerLayers[l]; ++i) {
			dEdS[l - 1][i] *= dReLU(sumNeurons[l - 1][i]);
		}
	}

	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			biasGradientAverage[l][j] += dEdS[l][j];
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				weightsGradientAverage[l][i][j] += neurons[l][i] * dEdS[l][j];
			}
			dEdS[l][j] = 0;
		}
	}

	for (int l = layers - 2; l >= 0; --l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			sumNeurons[l][j] = 0;
		}
	}

}

template<typename T_1>
void SCNN<T_1>::adaptiveLR() {
	T_1 tmp = 0;
	++iteratorALR;
	CEAverage += CE;
	CEQueue.emplace(CE);

	if (!flag_learning && iteratorALR >= ALRConstAv) {
		flag_learning = true;
		cout << "CEAverage: " << CEAverage << endl;
	}

	if (flag_learning) {
		tmp = CEAverage / ALRConstAv;
		CEDifferenceAv += tmp - CE;
		CEDifferenceLast.emplace(tmp - CE);
		if (iteratorALR >= ALRConst + ALRConstAv) flag_learningActivate = true;
		if (flag_learningActivate) {
			cout << "iteratorALR: " << iteratorALR << "\tCEAverage: " << CEAverage / ALRConstAv << "\tCEDifferenceAv : " << CEDifferenceAv / ALRConst << endl;
			if (iteratorALR >= ALRConstAv && CEDifferenceAv / ALRConst <= 0 + tmp / (2 * ALRConst) && CEDifferenceAv / ALRConst >= 0 - tmp / (2 * ALRConst)) {
				iteratorALR = 0;
				lamda *= Beta + 0.05;//0.999;
				Alfa *= Beta;
				CEAvLast = CEAverage;
				cout << "CEDifferenceAV: " << CEDifferenceAv / ALRConst << "\tCEAverage: " << CEAverage << "\tCEDifLast: " << CEDifferenceLast.front()
					<< "\tlamda: " << lamda << "\tAlfa: " << Alfa << endl;
			}
			CEDifferenceAv -= CEDifferenceLast.front();
			CEDifferenceLast.pop();
		}
		CEAverage -= CEQueue.front();
		CEQueue.pop();
	}
}

template<typename T_1>
void SCNN<T_1>::SGD() {
	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();

	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			bias[l][j] = bias[l][j] - lamda * biasGradientAverage[l][j] / backPropIterator;
			biasGradientAverage[l][j] = 0;
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				weights[l][i][j] = weights[l][i][j] - lamda * weightsGradientAverage[l][i][j] / backPropIterator;
				weightsGradientAverage[l][i][j] = 0;
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::Momentum() {
	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();

	if (backPropIterator % miniBatch == 0) {
		for (int l = 0; l < layers - 1; ++l) {
			for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
				momentumBias[l][j] = Beta * momentumBias[l][j] + I_Beta * (biasGradientAverage[l][j] / backPropIterator);
				bias[l][j] = bias[l][j] - Alfa * momentumBias[l][j];
				biasGradientAverage[l][j] = 0;
				for (int i = 0; i < sizePerLayers[l]; ++i) {
					momentumWeights[l][i][j] = Beta * momentumWeights[l][i][j] + I_Beta * weightsGradientAverage[l][i][j] / backPropIterator;
					weights[l][i][j] = weights[l][i][j] - Alfa * momentumWeights[l][i][j];
					weightsGradientAverage[l][i][j] = 0;
				}
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::NesterovMomentum() {
	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();

	if (backPropIterator % miniBatch == 0) {
		for (int l = 0; l < layers - 1; ++l) {
			for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
				momentumBias[l][j] = Beta * momentumBias[l][j] + I_Beta * biasGradientAverage[l][j] / backPropIterator;
				bias[l][j] = bias[l][j] - Alfa * momentumBias[l][j];
				biasGradientAverage[l][j] = 0;
				for (int i = 0; i < sizePerLayers[l]; ++i) {
					momentumWeights[l][i][j] = Beta * momentumWeights[l][i][j] + I_Beta * weightsGradientAverage[l][i][j] / backPropIterator;
					weights[l][i][j] = weights[l][i][j] - Alfa * momentumWeights[l][i][j];
					weightsGradientAverage[l][i][j] = 0;
				}
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::RMSprop() {
	T_1 tmp = 0;
	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();

	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			RMSBias[l][j] = Beta * RMSBias[l][j] + I_Beta * pow(biasGradientAverage[l][j] / backPropIterator, 2);;
			bias[l][j] = bias[l][j] - lamda * biasGradientAverage[l][j] / (backPropIterator * sqrt(RMSBias[l][j]) + de);
			biasGradientAverage[l][j] = 0;
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				RMSWeights[l][i][j] = Beta * RMSWeights[l][i][j] + I_Beta * pow(weightsGradientAverage[l][i][j] / backPropIterator, 2);
				weights[l][i][j] = weights[l][i][j] - lamda * weightsGradientAverage[l][i][j] / (backPropIterator * sqrt(RMSWeights[l][i][j]) + de);
				weightsGradientAverage[l][i][j] = 0;
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::Adam() {
	T_1 tmp_bias = 0;
	T_1 tmp_weights = 0;

	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();
	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			tmp_bias = biasGradientAverage[l][j] / backPropIterator;
			RMSBias[l][j] = Beta * RMSBias[l][j] + I_Beta * pow(tmp_bias, 2);
			momentumBias[l][j] = Beta * momentumBias[l][j] + I_Beta * tmp_bias;
			bias[l][j] = bias[l][j] - Alfa * momentumBias[l][j] / (backPropIterator * sqrt(RMSBias[l][j]) + de);
			biasGradientAverage[l][j] = 0;
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				tmp_weights = weightsGradientAverage[l][i][j] / backPropIterator;
				RMSWeights[l][i][j] = Beta * RMSWeights[l][i][j] + I_Beta * pow(tmp_weights, 2);
				momentumWeights[l][i][j] = Beta * momentumWeights[l][i][j] + I_Beta * tmp_weights;
				weights[l][i][j] = weights[l][i][j] - Alfa * momentumWeights[l][i][j] / (backPropIterator * sqrt(RMSWeights[l][i][j]) + de);
				weightsGradientAverage[l][i][j] = 0;
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::AdamNM() {
	T_1 tmp_bias = 0;
	T_1 tmp_weights = 0;

	CE /= backPropIterator;
	errorOutput();
	//adaptiveLR();
	for (int l = 0; l < layers - 1; ++l) {
		for (int j = 0; j < sizePerLayers[l + 1]; ++j) {
			tmp_bias = biasGradientAverage[l][j] / backPropIterator;
			RMSBias[l][j] = Beta * RMSBias[l][j] + I_Beta * pow(tmp_bias, 2);
			momentumBias[l][j] = Beta * momentumBias[l][j] + I_Beta * tmp_bias;
			bias[l][j] = bias[l][j] - Alfa * momentumBias[l][j] / (backPropIterator * sqrt(RMSBias[l][j]) + de);
			biasGradientAverage[l][j] = 0;
			for (int i = 0; i < sizePerLayers[l]; ++i) {
				tmp_weights = weightsGradientAverage[l][i][j] / backPropIterator;
				RMSWeights[l][i][j] = Beta * RMSWeights[l][i][j] + I_Beta * pow(tmp_weights, 2);
				momentumWeights[l][i][j] = Beta * momentumWeights[l][i][j] + I_Beta * tmp_weights;
				weights[l][i][j] = weights[l][i][j] - Alfa * momentumWeights[l][i][j] / (backPropIterator * sqrt(RMSWeights[l][i][j]) + de);
				weightsGradientAverage[l][i][j] = 0;
			}
		}
	}
	backPropIterator = 0;
}

template<typename T_1>
void SCNN<T_1>::learning(int epoch, SCNNMethod methods) {
	int iter_learning = 0;
	numberEpoch = epoch;
	ofstream fout;
	switch (methods) {
	case SCNNMethod::SGDMethod:
		cout << "SGD activate\n";
		//fout.open("Result_SGD_WLR.txt");
		//if (!fout.is_open()) {
			//system("pause");
			//cout << "ERROR!!! File:Result_SGD_WLR.txt can not be open!!!\n";
		//}
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					SGD();
					//fout << NNIterator << "\t" << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		//fout.close();
		break;
	case SCNNMethod::MOMENTUM:
		cout << "MOMENTUM activate\n";
		fout.open("Result_MOMENTUM_WLR.txt");
		if (!fout.is_open()) system("pause");
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			CEPerEpochTMP = 0;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					Momentum();
					fout << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case SCNNMethod::NESTEROVMOMENTUM:
		cout << "NESTEROV MOMENTUM activate\n";
		fout.open("Result_NesterovMOMENTUM_WLR.txt");
		if (!fout.is_open()) system("pause");
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functionNMdEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					NesterovMomentum();
					fout << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case SCNNMethod::RMSPROP:
		cout << "RMSPROP activate\n";
		fout.open("Result_RMS_WLR.txt");
		if (!fout.is_open()) system("pause");
		setVectorRMS();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					RMSprop();
					fout << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case SCNNMethod::ADAM:
		cout << "ADAM ACTIVATE\n";
		fout.open("Result_ADAM_WLR.txt");
		if (!fout.is_open()) system("pause");
		setVectorRMS();
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					Adam();
					fout << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case SCNNMethod::ADAMNM:
		cout << "ADAMNM activate\n";
		cout << "ADAM NM is actevated\n";
		fout.open("Result_NesterovADAM_WLR.txt");
		if (!fout.is_open()) {
			cout << "ERROR OPENED FILE\n";
			system("pause");
		}
		setVectorRMS();
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			//fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functionNMdEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					AdamNM();
					fout << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	}
}

template<typename T_1>
void SCNN<T_1>::learning(int epoch) {
	cout << "Learning exper has activated\n";
	int iter_learning = 0;
	numberEpoch = epoch;

	for (int epo = 0; epo < epoch; ++epo) {
		Epoch = epo;
		CEPerEpoch = CEPerEpochTMP / examplsNumber;
		CEPerEpochTMP = 0;
		//fout << "CE per epoch: " << CEPerEpoch << endl;
		for (int i = 0; i < examplsNumber; ++i) {
			for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
			originalData[dataSetAll[i][0]] = 1;
			for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
			directPass();
			functiondEdS();
			if (backPropIterator % miniBatch == 0) {
				CEPerEpochTMP += CE;
				SGD();
				//fout << NNIterator << "\t" << CE << endl;
				CE = 0;
			}
			++NNIterator;
		}
	}
}

/*template<typename T_1>
void SCNN<T_1>::learning(int epoch, int iter_tmp) {
	int iter_learning = 0;
	numberEpoch = epoch;
	ofstream fout;
	switch (iter_tmp) {
	case 0:
		cout << "SGD is actevated\n";
		fout.open("Result_SGD_WLR.txt");
		if (!fout.is_open()) system("pause");
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					SGD();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case 1:
		cout << "Momentum is actevated\n";
		fout.open("Result_MOMENTUM.txt");
		if (!fout.is_open()) system("pause");
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			CEPerEpochTMP = 0;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					Momentum();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case 2:
		cout << "Nesterov M is actevated\n";
		fout.open("Result_NesterovMOMENTUM.txt");
		if (!fout.is_open()) system("pause");
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functionNMdEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					NesterovMomentum();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case 3:
		cout << "RMS is actevated\n";
		fout.open("Result_RMS.txt");
		if (!fout.is_open()) system("pause");
		setVectorRMS();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					RMSprop();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case 4:
		cout << "ADAM  is actevated\n";
		fout.open("Result_ADAM.txt");
		if (!fout.is_open()) system("pause");
		setVectorRMS();
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functiondEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					Adam();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	case 5:
		cout << "ADAM NM is actevated\n";
		fout.open("Result_NesterovADAM.txt");
		if (!fout.is_open()) {
			cout << "ERROR OPENED FILE\n";
			system("pause");
		}
		setVectorRMS();
		setVectorMomentum();
		for (int epo = 0; epo < epoch; ++epo) {
			Epoch = epo;
			CEPerEpoch = CEPerEpochTMP / examplsNumber;
			CEPerEpochTMP = 0;
			fout << "CE per epoch: " << CEPerEpoch << endl;
			for (int i = 0; i < examplsNumber; ++i) {
				for (int j = 0; j < sizePerLayers[layers - 1]; ++j) originalData[j] = 0;
				originalData[dataSetAll[i][0]] = 1;
				for (int j = 0; j < sizePerLayers[0]; ++j) neurons[0][j] = dataSetAll[i][j + 1];
				directPass();
				functionNMdEdS();
				if (backPropIterator % miniBatch == 0) {
					CEPerEpochTMP += CE;
					AdamNM();
					fout << "iteration: " << NNIterator << "\tCE: " << CE << endl;
					CE = 0;
				}
				++NNIterator;
			}
		}
		fout.close();
		break;
	}
}
*/

template<typename T_1>
void SCNN<T_1>::readDataAll(string fileName) {
#ifdef DEBUG
	cout << "file name: " << fileName << endl;
#endif // DEBUG

	ifstream fin;
	fin.open(fileName, ios_base::binary);
	while (!fin.is_open()) {
		cout << "ERROR. File " << fileName << "can not by opened\n";
		cout << "Enter a file name" << endl;
		cin >> fileName;
		fin.open(fileName);
	}
	fin.read((char*)&examplsNumber, sizeof(examplsNumber));
	dataSetAll = vector<vector<T_1>>(examplsNumber);
	for (int i = 0; i < examplsNumber; ++i) {
		dataSetAll[i] = vector<T_1>(pixelsPerImage + 1, 0);
	}
	for (int i = 0; i < examplsNumber; ++i) {
		fin.read((char*)&dataSetAll[i][0], sizeof(dataSetAll[i][0]));
		for (int j = 1; j < pixelsPerImage + 1; ++j) {
			fin.read((char*)&dataSetAll[i][j], sizeof(dataSetAll[i][j]));
		}
	}
#ifdef DEBUG_READ_DATA_ALL
	cout << "examplsNumber: " << examplsNumber << endl;
	for (int i = examplsNumber - 100; i < examplsNumber; ++i) {
		for (int j = 0; j < pixelsPerImage + 1; ++j) {
			cout << "dataSetAll[" << i << "][" << j << "]: " << dataSetAll[i][j] << "\t";
			if (j % 3 == 0) cout << endl;
		}
	}
#endif // DEBUG

	fin.close();
}

template<typename T_1>
void SCNN<T_1>::readData(string fileName) {
	T_1 tmp = 0;
	ifstream fin;
	fin.open(fileName, ios_base::binary);
	while (!fin.is_open()) {
		cout << "ERROR. File " << fileName << "can not be opened\n";
		cout << "Enter a file name" << endl;
		cin >> fileName;
		fin.open(fileName);
	}
	if (mooving == 0) {
		fin.read((char*)&examplsNumber, sizeof(examplsNumber));
		mooving += sizeof(examplsNumber);
#ifdef DEBUG_READ_DATA
		cout << "examplNumber: " << examplsNumber << endl;
#endif // DEBUG
	}
	fin.seekg(mooving, ios_base::beg);
	fin.read((char*)&tmp, sizeof(tmp));
#ifdef DEBUG_READ_DATA
	cout << "tmp: " << tmp << endl;
#endif // DEBUG
	
	for (int i = 0; i < sizePerLayers[layers - 1]; ++i) originalData[i] = 0;
	originalData[tmp] = 1;

	for (int i = 0; i < pixelsPerImage; ++i) {
		fin.read((char*)&neurons[0][i], sizeof(neurons[0][i]));
#ifdef DEBUG_READ_DATA
		if (iterationReadData > -1) {
			cout << "neurons[" << iterationReadData << "][" << i << "]: " << neurons[0][i] << "\t";
			if (i % 3 == 0) cout << endl;
		}
#endif // DEBUG
	}

	mooving += sizeof(tmp) + pixelsPerImage * sizeof(neurons[0][0]);
	++iterationReadData;
	if (iterationReadData >= examplsNumber) {
		mooving = 0;
		iterationReadData = 0;
	}
	fin.close();
}

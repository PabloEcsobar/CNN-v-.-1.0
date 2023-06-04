#include <thread>
#include "SCNN.h"


int main() {
	int layers = 3;
	int epoch = 3;
	int numberThread = 2;
	int examplsNumber = 0;
	float tmp = 0;
	string fileName = "lib_MNIST_edit_binary.bin";
	vector<int> siazePerLayers(layers);
	for (int i = 0; i < layers; ++i) {
		cout << "Enter size for " << i << " layers: ";
		cin >> siazePerLayers[i];
	}
	ifstream fin;
	fin.open(fileName, ios_base::binary);
	if (!fin.is_open()) {
		cout << "ERROR!!!";
		system("pause");
	}
	fin.read((char*)&tmp, sizeof(tmp));
	fin.close();
	examplsNumber = int(tmp);
	cout << "exampls number: " << examplsNumber << endl;
	system("pause");


	SCNN<float> obj_1(layers, siazePerLayers);
	SCNN<float> obj_2(layers, siazePerLayers);
	SCNN<float> obj_3(layers, siazePerLayers);
	SCNN<float> obj_4(layers, siazePerLayers);
	SCNN<float> obj_5(layers, siazePerLayers);
	SCNN<float> obj_6(layers, siazePerLayers);

	obj_1.readDataAll(fileName);
	obj_2.readDataAll(fileName);
	obj_3.readDataAll(fileName);
	obj_4.readDataAll(fileName);
	obj_5.readDataAll(fileName);
	obj_6.readDataAll(fileName);

	thread th_1([&]() {obj_1.learning(epoch, SCNN<float>::ADAM); });
	thread th_2([&]() {obj_2.learning(epoch, SCNN<float>::ADAMNM); });
	thread th_3([&]() {obj_3.learning(epoch, SCNN<float>::MOMENTUM); });
	thread th_4([&]() {obj_4.learning(epoch, SCNN<float>::NESTEROVMOMENTUM); });
	thread th_5([&]() {obj_5.learning(epoch, SCNN<float>::RMSPROP); });
	thread th_6([&]() {obj_6.learning(epoch, SCNN<float>::SGDMethod); });


	th_1.join();
	th_2.join();
	th_3.join();
	th_4.join();
	th_5.join();
	th_6.join();
	

	/*SCNN<float>* obj_1 = new SCNN<float>[numberThread];
	thread* th_ptr = new thread[numberThread];
	for (int i = 0; i < numberThread; ++i) {
		obj_1[i] = SCNN<float>{ layers, siazePerLayers };
		obj_1[i].readDataAll(fileName);
	}
	for (int i = 0; i < numberThread; ++i) th_ptr[i] = thread{ [&]() {obj_1[i].learning(epoch); } };

	for (int i = 0; i < numberThread; ++i) 	th_ptr[i].join();
	*/

	system("pause");
	return 0;
}
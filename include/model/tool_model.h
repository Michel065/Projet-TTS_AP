#include <iostream>
#include <string>
class Model {

public:
	Model(std::string model_name);

	void print();

private:
	std::string _model_name;
};
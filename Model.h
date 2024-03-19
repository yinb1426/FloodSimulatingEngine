#pragma once
#include <vector>
using namespace std;

class Model
{
protected:
	unsigned int sizeX;
	unsigned int sizeY;
	double deltaT;
	vector<double> terrainHeight;
	vector<double> waterHeight;
	double* gTerrainHeight;
	double* gWaterHeight;

public:
	Model(unsigned int _sizeX, unsigned int _sizeY, double _deltaT, vector<double> _terrainHeight, vector<double> _waterHeight)
		: sizeX(_sizeX), sizeY(_sizeY), deltaT(_deltaT), terrainHeight(_terrainHeight), waterHeight(_waterHeight)
	{}

	~Model() {}

	void SetSizeX(const unsigned int sizeX);
	unsigned int GetSizeX() const;
	void SetSizeY(const unsigned int sizeY);
	unsigned int GetSizeY() const;
	void SetDeltaT(const double deltaT);
	double GetDeltaT() const;
	void SetTerrainHeight(const vector<double> terrainHeight);
	vector<double> GetTerrainHeight() const;
	void SetWaterHeight(const vector<double> waterHeight);
	vector<double> GetWaterHeight() const;

public:
	virtual void InitDevice() = 0;
	virtual void SendAllDataToDevice() = 0;
	virtual void PreparationForSimulaion() = 0;
	virtual void RunSimulation(const unsigned int step) = 0;
	virtual void GetResultFromDevice() = 0;
	virtual void FreeAllData() = 0;
};


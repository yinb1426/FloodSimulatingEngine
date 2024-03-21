#pragma once
#include "Dam.h"
#include "Drain.h"
#include "Model.h"
#include <iostream>

class Project
{
protected:
	shared_ptr<Model> model;
	std::vector<Dam> damList;
	std::vector<Drain> drainList;

public:
	Project(shared_ptr<Model> _model) : model(_model), damList(std::vector<Dam>(0)), drainList(std::vector<Drain>(0)) {}
	~Project() {}

public:
	void AddDam(const Dam d);
	Dam GetDam(const size_t index) const;
	void ClearDamList();

	void AddDrain(const Drain d);
	Drain GetDrain(const size_t index) const;
	void ClearDrainList();

	shared_ptr<Model> GetModel() const;

	void Preparation();
	void RunSimulation(const size_t steps);
	void RunSimulationOneStep(const size_t step);
	vector<double> GetWaterHeight();
	void GetResult();
	void Ending();

};
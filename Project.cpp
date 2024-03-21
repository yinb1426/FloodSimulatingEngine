#include "Project.h"

void Project::AddDam(const Dam d)
{
	damList.push_back(d);
}

Dam Project::GetDam(const size_t index) const
{
	if (index < 0 || index >= damList.size())
		throw("Index Wrong!");
	return damList[index];
}

void Project::ClearDamList()
{
	damList.clear();
}

void Project::AddDrain(const Drain d)
{
	drainList.push_back(d);
}

Drain Project::GetDrain(const size_t index) const
{
	if (index < 0 || index >= drainList.size())
		throw("Index Wrong!");
	return drainList[index];
}

void Project::ClearDrainList()
{
	drainList.clear();
}

shared_ptr<Model> Project::GetModel() const
{
	return shared_ptr<Model>(model);
}

void Project::Preparation()
{
	model->InitDevice();
	model->SendAllDataToDevice();
	model->PreparationForSimulaion();
}

void Project::RunSimulation(const size_t steps)
{
	size_t totalSteps = steps;
	if (steps < 0)
		totalSteps = std::numeric_limits<size_t>::max();
	size_t step = 0;
	while (step < totalSteps)
	{
		model->RunSimulation(step);
		++step;
	}
}

void Project::RunSimulationOneStep(const size_t step)
{
	if (step < 0)
		throw("Step Wrong!");
	model->RunSimulation(step);
}

vector<double> Project::GetWaterHeight()
{
	return model->GetWaterHeight();
}

void Project::GetResult()
{
	model->GetResultFromDevice();
}

void Project::Ending()
{
	model->FreeAllData();
}

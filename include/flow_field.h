#pragma once
#include <string>
class PINNModel;

// Generate and save a flow field grid to CSV
void generateFlowField(PINNModel& model, const std::string& filename);


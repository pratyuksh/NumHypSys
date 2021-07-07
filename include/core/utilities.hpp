#ifndef CORE_UTILITIES_HPP
#define CORE_UTILITIES_HPP

#include <iostream>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>

#include "config.hpp"


//! Writes convergence results to json file
void write_convg_json_file (std::string system_type,
                            const nlohmann::json& config,
                            Eigen::VectorXd &data_x,
                            Eigen::MatrixXd &data_y);


#endif /// CORE_UTILITIES_HPP

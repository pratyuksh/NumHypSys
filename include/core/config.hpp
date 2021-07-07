#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <nlohmann/json.hpp>

//! Creates an object of the nlohmann json library
//! to read json file
//! passed through the command-line arguments
nlohmann::json get_global_config(int argc,
                                 char* const argv[]);

//! Creates an object of the nlohmann json library
//! to read fileName.json
nlohmann::json get_global_config(const std::string& fileName);

#endif /// CONFIG_HPP

#include <../include/core/config.hpp>
#include <iostream>
#include <fstream>


//! Creates an object of the nlohmann json library
//! to read json file
//! passed through the command-line arguments
nlohmann::json get_global_config(int argc, char* const argv[])
{
    nlohmann::json config; // empty
    if (argc == 2) 
    {
        std::string fileName = argv[1];
		config = get_global_config (fileName);
	}
	else {
		std::cout << "\nError: Wrong number of arguments.\n";
	}
	
	return config;
}

//! Creates an object of the nlohmann json library
//! to read fileName.json
nlohmann::json get_global_config(const std::string& fileName)
{
    std::ifstream file(fileName);
    nlohmann::json config;
    file >> config;
    return config;
}

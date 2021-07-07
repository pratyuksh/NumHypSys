#ifndef UQ_GENERATOR_HPP
#define UQ_GENERATOR_HPP

#include <random>


//! Class Generator
//! generates random numbers for specified distributions
//! using Mersenne-Twister generator
template<class Distribution>
class Generator
{
public:
    //! Constructors
    explicit Generator ();
    explicit Generator (const int seed);

    //! Returns a random number from the generator
    double operator() (Distribution&);
    
private:
    std::mt19937 m_mtgen;
};


//! Generator explicit instantiation for Uniform distribution
template
class Generator < std::uniform_real_distribution<double> >;


#endif /// UQ_GENERATOR_HPP

#include "../../../include/uq/sampler/generator.hpp"
#include <ctime>


//! Constructors
template <class Distribution>
Generator <Distribution>
:: Generator ()
{
    const int SEED = 0; // fixed
    //const int SEED = time(nullptr);
    m_mtgen.seed(SEED);
}

template <class Distribution>
Generator <Distribution>
:: Generator (const int seed)
{
    m_mtgen.seed(seed);
}

//! Returns a random number from the generator
template <class Distribution>
double Generator <Distribution>
:: operator() (Distribution& dist)
{
    return dist(m_mtgen);
}


// End of file

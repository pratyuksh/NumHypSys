#ifndef UQ_SAMPLER_HPP
#define UQ_SAMPLER_HPP

#include <random>
#include <cmath>

#include "../../core/config.hpp"
#include "../../../include/uq/sampler/generator.hpp"
#include "mfem.hpp"

using namespace mfem;


// Sampler types
enum {Uniform,
      Sobol};


//! Base template class for Sampler
template<int SamplerType>
class Sampler;


//! Template specialization
//! for Uniform distribution between [-1, +1]
template<>
class Sampler <Uniform>
{
public:
    //! Constructors
    explicit Sampler (const nlohmann::json& config);
    explicit Sampler (const int nparams,
                      const int nsamples);
    explicit Sampler (const int seed,
                      const int nparams,
                      const int nsamples);

    //! Generates one sample
    Vector generate_one_sample();

    //! Generates all samples
    DenseMatrix generate();
    
private:
    std::uniform_real_distribution<double> m_unif;
    Generator<std::uniform_real_distribution<double>> m_gen;
    int m_nparams, m_nsamples;
};


//! Template specialization
//! for Sobol points
/*template<>
class Sampler <Sobol>
{
public:
    //! Constructors
    explicit Sampler (const nlohmann::json& config);
    explicit Sampler (const int nparams,
                      const int nsamples);
    explicit Sampler (const int seed,
                      const int nparams,
                      const int nsamples);

    //! Generates all samples
    DenseMatrix generate();

private:
    int m_seed;
    int m_nparams, m_nsamples;
};*/


//! Makes samples
//! by scattering the samples from root
//! to all the other processors
std::pair<DenseMatrix, Array<int>>
make_samples (MPI_Comm&,
              std::string samplerType,
              int, int, Array<int>,
              int seed=0);

std::pair<DenseMatrix, Array<int>>
make_samples (MPI_Comm&, MPI_Comm&,
              std::string samplerType,
              int, int, Array<int>,
              int seed=0);

//! Makes sample Ids
//! by scattering the sample Ids from root
//! to all the other processors
Array<int> make_sampleIds (MPI_Comm&, int, Array<int>);


#endif /// UQ_SAMPLER_HPP

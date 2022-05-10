// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <vector>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions

#include <dune/local_integration_test/local_integration_test.hh>

int main(int argc, char** argv)
{
  try{
    
    using Scalar = double;
    constexpr size_t dim = 3;
    constexpr size_t number_of_dofs = 4;
    using Vector = Dune::FieldVector<Scalar, dim>;
  
    Scalar distance_dipole = 4.0;
    Scalar edgeLength = 1.0;
    Scalar distance_coil = 10.0;
    size_t intorder_eeg = 5;
    size_t intorder_meg_inaccurate = 3;
    size_t intorder_meg_accurate = 50;
  
    // Maybe initialize MPI
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

    // create mesh
    Scalar csf_conductivity = 1.79; // S/m
    Scalar grey_matter_conductivity = 0.33; // S/m
    duneuro::SingleTetrahedronMesh<Scalar> mesh(edgeLength, csf_conductivity, grey_matter_conductivity);
    mesh.bindDipole({distance_dipole * 0.793206996070584536, distance_dipole * -0.608210651451048467, distance_dipole * 0.030041052680805216}, 
                    {-0.803530014579211693, 0.359998305533424823, -0.474068281667730462});
    mesh.bindCoil({distance_coil * -0.513372253148087432, distance_coil * -0.858054442607775303, distance_coil * -0.013838468799547596});
    
    const auto& element = mesh.entity();
    const auto& intersection = mesh.frontFacingIntersection();
    
    std::cout << " EEG : \n";
    
    std::vector<Scalar> numericIntegrals = mesh.numericSurfaceIntegrals(intorder_eeg);
    std::cout << " Numeric integrals with intorder " << intorder_eeg << " : \n";
    for(size_t i = 0; i < number_of_dofs; ++i) {
      std::cout << numericIntegrals[i] << "\t";
    }
    
    std::vector<Scalar> analyticIntegrals = mesh.analyticSurfaceIntegrals();
    std::cout << "\n Analytic integrals : \n";
    for(size_t i = 0; i < number_of_dofs; ++i) {
      std::cout << analyticIntegrals[i] << "\t";
    }
    
    Scalar norm_analytic(0.0);
    Scalar norm_diff(0.0);
    for(size_t i = 0; i < number_of_dofs; ++i) {
      norm_diff += (numericIntegrals[i] - analyticIntegrals[i]) * (numericIntegrals[i] - analyticIntegrals[i]);
      norm_analytic += analyticIntegrals[i] * analyticIntegrals[i];
    }
    norm_analytic = std::sqrt(norm_analytic);
    norm_diff = std::sqrt(norm_diff);
    Scalar relative_error = norm_diff / norm_analytic;
    
    std::cout << "\n Relativ error : " << relative_error << "\n";
    
    std::cout << "\n\n\n";
    
    std::cout << " MEG : \n";
    //auto magnetic_field_inaccurate = mesh.numericSurfaceMagneticField(intorder_meg_inaccurate);
    //auto magnetic_field_accurate = mesh.numericSurfaceMagneticField(intorder_meg_accurate);
    
    //auto magnetic_field_inaccurate = mesh.numericPatchMagneticField(intorder_meg_inaccurate);
    //auto magnetic_field_accurate = mesh.numericPatchMagneticField(intorder_meg_accurate);
    
    auto magnetic_field_inaccurate = mesh.numericTransitionMagneticField(intorder_meg_inaccurate);
    auto magnetic_field_accurate = mesh.numericTransitionMagneticField(intorder_meg_accurate);
    
    std::cout << " Inacurate field with intorder " << intorder_meg_inaccurate << ": \n" << magnetic_field_inaccurate << "\n";
    std::cout << " Accurate field with intorder " << intorder_meg_accurate << ": \n" << magnetic_field_accurate << "\n"; 
    
    Scalar norm_accurate(0.0);
    Scalar norm_meg_diff(0.0);
    for(size_t i = 0; i < dim; ++i) {
      norm_meg_diff += (magnetic_field_accurate[i] - magnetic_field_inaccurate[i]) * (magnetic_field_accurate[i] - magnetic_field_inaccurate[i]);
      norm_accurate += magnetic_field_accurate[i] * magnetic_field_accurate[i];
    }
    norm_accurate = std::sqrt(norm_accurate);
    norm_meg_diff = std::sqrt(norm_meg_diff);
    Scalar relative_error_meg = norm_meg_diff / norm_accurate;
    
    std::cout << "\n Relativ error : " << relative_error_meg << "\n";
    
    std::cout << "\n The program didn't crash!\n";
    return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}

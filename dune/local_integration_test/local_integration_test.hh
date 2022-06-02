#ifndef LOCAL_INTEGRATION_TEST_HH
#define LOCAL_INTEGRATION_TEST_HH

#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>

#include <dune/geometry/type.hh>
#include <dune/grid/uggrid.hh>
#include <dune/localfunctions/lagrange.hh>
#include <dune/pdelab.hh>
#include <dune/pdelab/common/crossproduct.hh>

#include <duneuro/eeg/subtraction_dg_uinfty.hh>
#include <duneuro/eeg/analytic_utilities.hh>
#include <duneuro/common/flags.hh>

namespace duneuro {

  struct SingleElementMeshTraits {
    enum {dim = 3};
    using Grid = Dune::UGGrid<dim>;
    using Factory = Dune::GridFactory<Grid>;
    using Scalar = double;
    enum {ansatzfunction_degree = 1};
  };

  template<ElementType elementType>
  struct ElementTypeSpecificTraits;

  template<>
  struct ElementTypeSpecificTraits<ElementType::tetrahedron> {
    using FiniteElement = Dune::LagrangeSimplexLocalFiniteElement<SingleElementMeshTraits::Scalar,
                                                                  SingleElementMeshTraits::Scalar,
                                                                  SingleElementMeshTraits::dim,
                                                                  SingleElementMeshTraits::ansatzfunction_degree>;
    enum {number_of_dofs = 4};
    enum {number_of_facet_corners = 3};
  };

  template<>
  struct ElementTypeSpecificTraits<ElementType::hexahedron> {
    using FiniteElement = Dune::LagrangeCubeLocalFiniteElement<SingleElementMeshTraits::Scalar,
                                                               SingleElementMeshTraits::Scalar,
                                                               SingleElementMeshTraits::dim,
                                                               SingleElementMeshTraits::ansatzfunction_degree>;
    enum {number_of_dofs = 8};
    enum {number_of_facet_corners = 4};
  };

  template<ElementType elementType>
  void insertSingleElement(SingleElementMeshTraits::Factory&, SingleElementMeshTraits::Scalar);

  template<>
  void insertSingleElement<ElementType::tetrahedron>(SingleElementMeshTraits::Factory& factory, SingleElementMeshTraits::Scalar edgeLength) {
    factory.insertVertex({0.0                                 ,   - edgeLength / 2.0, (- std::sqrt(3) / 6.0) * edgeLength});
    factory.insertVertex({0.0                                 ,     edgeLength / 2.0, (- std::sqrt(3) / 6.0) * edgeLength});
    factory.insertVertex({0.0                                 ,     0.0              , (  std::sqrt(3) / 3.0) * edgeLength});
    factory.insertVertex({(- std::sqrt(6) / 3.0) * edgeLength,     0.0              ,    0.0});

    factory.insertElement(Dune::GeometryTypes::simplex(SingleElementMeshTraits::dim), {0, 1, 2, 3});
    factory.insertBoundarySegment({0, 1, 2});
    return;
  }

  template<>
  void insertSingleElement<ElementType::hexahedron>(SingleElementMeshTraits::Factory& factory, SingleElementMeshTraits::Scalar edgeLength) {
    factory.insertVertex({-1.0 * edgeLength, -0.5 * edgeLength, -0.5 * edgeLength});
    factory.insertVertex({ 0.0 * edgeLength, -0.5 * edgeLength, -0.5 * edgeLength});
    factory.insertVertex({-1.0 * edgeLength,  0.5 * edgeLength, -0.5 * edgeLength});
    factory.insertVertex({ 0.0 * edgeLength,  0.5 * edgeLength, -0.5 * edgeLength});
    factory.insertVertex({-1.0 * edgeLength, -0.5 * edgeLength,  0.5 * edgeLength});
    factory.insertVertex({ 0.0 * edgeLength, -0.5 * edgeLength,  0.5 * edgeLength});
    factory.insertVertex({-1.0 * edgeLength,  0.5 * edgeLength,  0.5 * edgeLength});
    factory.insertVertex({ 0.0 * edgeLength,  0.5 * edgeLength,  0.5 * edgeLength});

    factory.insertElement(Dune::GeometryTypes::cube(SingleElementMeshTraits::dim), {0, 1, 2, 3, 4, 5, 6, 7});
    factory.insertBoundarySegment({1, 3, 5, 7});
  }

  template <ElementType elementType>
  class SingleElementMesh {
  public:
    // typedefs
    using Scalar = double;
    enum {dim = SingleElementMeshTraits::dim};
    using Traits = ElementTypeSpecificTraits<elementType>;
    enum {number_of_dofs = Traits::number_of_dofs};
    enum {number_of_facet_corners = Traits::number_of_facet_corners};
    using Grid = typename SingleElementMeshTraits::Grid;
    using Factory = typename SingleElementMeshTraits::Factory;
    using GridView = typename Grid::LeafGridView;
    using Entity = typename GridView::template Codim<0>::Entity;
    using Intersection = typename GridView::Intersection;
    using UInfinity = InfinityPotential<GridView, Scalar>;
    using UInfinityGradient = InfinityPotentialGradient<GridView, Scalar>;
    using Vector = Dune::FieldVector<Scalar, dim>;
    enum {degree = 1};
    using FiniteElement = typename Traits::FiniteElement;
    using Tensor = Dune::FieldMatrix<Scalar, dim, dim>;
    
  
    SingleElementMesh(Scalar edgeLength, Scalar sigma, Scalar sigma_infinity)
      : edgeLength_(edgeLength)
      , sigma_(0.0)
      , sigma_infinity_(0.0)
      , gridPtr_(nullptr)
      , dof_to_vertex_indices_(number_of_dofs)
      , vertex_to_dof_indices_(number_of_dofs)
      , frontIntersectionIndices_(number_of_facet_corners)
      , fem_()
      , corners_(number_of_dofs)
      , frontFacingNormal_(0.0)
      , dipole_position_({1.0, 0.0, 0.0})
      , dipole_moment_({1.0, 0.0, 0.0})
      , coil_position_({2.0 * edgeLength, 2 * edgeLength, 0.0})
    {
      // create grid
      Factory factory;
      insertSingleElement<elementType>(factory, edgeLength_);
      gridPtr_ = factory.createGrid();
      
      // get element
      GridView gridView = gridPtr_->leafGridView();
      entity_ = *(gridView.template begin<0>());
      auto geometry = entity_.geometry();
      
      for(size_t i = 0; i < dim; ++i) {
        sigma_[i][i] = sigma;
        sigma_infinity_[i][i] = sigma_infinity;
      }
      
      // get intersection in z-y plane
      for(const Intersection& intersection : Dune::intersections(gridView, entity_)) {
        if(factory.wasInserted(intersection)) {
          intersection_ = intersection;
          break;
        }
      }
      
      // create mapping between DOF and vertex indices
      for(size_t i = 0; i < number_of_dofs; ++i) {
        dof_to_vertex_indices_[i] = fem_.localCoefficients().localKey(i).subEntity();
        vertex_to_dof_indices_[dof_to_vertex_indices_[i]] = i;
        corners_[i] = geometry.corner(i);
      }
      
      // extract corner indices belonging to front facing intersection
      auto front_intersection_corner_iterator = Dune::referenceElement(entity_.geometry()).subEntities(intersection_.indexInInside(), 1, dim);
      std::copy(front_intersection_corner_iterator.begin(), front_intersection_corner_iterator.end(), frontIntersectionIndices_.begin());
      
      frontFacingNormal_ = intersection_.centerUnitOuterNormal();
    }
  
  void bindDipole(Vector dipole_position, Vector dipole_moment)
  {
    dipole_position_ = dipole_position;
    dipole_moment_ = dipole_moment;
  }
  
  void bindCoil(Vector coil_position)
  {
    coil_position_ = coil_position;
  }
  
  std::vector<Scalar> numericSurfaceIntegrals(size_t integration_order)
  {
    // create gradient of infinity potential with source at dipole_position_
    UInfinityGradient grad_u_infinity(gridPtr_->leafGridView());
    Tensor sigma_infinity_inverse(sigma_infinity_);
    sigma_infinity_inverse.invert();
    grad_u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    
    const auto& quad_rule = Dune::QuadratureRules<Scalar, dim - 1>::rule(intersection_.type(), integration_order);
    std::vector<Dune::FieldVector<Scalar, 1>> basis_vals(number_of_dofs);
    
    const auto& geometryInInside = intersection_.geometryInInside();
    const auto& geometryWorld = intersection_.geometry();
    
    std::vector<Scalar> integrals(number_of_dofs, 0.0);
    // perform numerical integration
    for(const auto& quad_point : quad_rule) {
      auto ref_triangle_coordinate = quad_point.position();
      Scalar factor = geometryWorld.integrationElement(ref_triangle_coordinate) * quad_point.weight();
      
      fem_.localBasis().evaluateFunction(geometryInInside.global(ref_triangle_coordinate), basis_vals);
      
      Vector vec;
      grad_u_infinity.evaluateGlobal(geometryWorld.global(ref_triangle_coordinate), vec);
      Vector sigma_infinity_grad_u_infinity;
      sigma_infinity_.mv(vec, sigma_infinity_grad_u_infinity);
      
      for(size_t i = 0; i < number_of_dofs; ++i) {
        integrals[dof_to_vertex_indices_[i]] += (sigma_infinity_grad_u_infinity * frontFacingNormal_) * basis_vals[i] * factor;
      }
    } // end loop over quadrature points
    
    return integrals;
  }
  
  std::vector<Scalar> numericPatchIntegrals(size_t integration_order)
  {
    // create gradient of infinity potential with source at dipole_position_
    UInfinityGradient grad_u_infinity(gridPtr_->leafGridView());
    Tensor sigma_infinity_inverse(sigma_infinity_);
    sigma_infinity_inverse.invert();
    grad_u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    
    Tensor sigma_corr = sigma_;
    sigma_corr -= sigma_infinity_;
    
    const auto& quad_rule = Dune::QuadratureRules<Scalar, dim>::rule(entity_.type(), integration_order);
    auto geometry = entity_.geometry();
    std::vector<Dune::FieldMatrix<Scalar, 1, dim>> basis_jacobians;
    
    std::vector<Scalar> integrals(number_of_dofs, 0.0);
    // perform numerical integration
    for(const auto& quad_point : quad_rule) {
      auto local_position = quad_point.position();
      Vector global_position = geometry.global(local_position);
      Scalar integration_factor = geometry.integrationElement(local_position) * quad_point.weight();
      
      Vector vec;
      grad_u_infinity.evaluateGlobal(global_position, vec);
      Vector sigma_corr_grad_u_infinity;
      sigma_corr.mv(vec, sigma_corr_grad_u_infinity);
      
      fem_.localBasis().evaluateJacobian(local_position, basis_jacobians);
      
      Tensor jacobian_inverse_transposed = geometry.jacobianInverseTransposed(local_position);
      Vector help;
      for(size_t i = 0; i < fem_.size(); ++i) {
        jacobian_inverse_transposed.mv(basis_jacobians[i][0], help);
        integrals[dof_to_vertex_indices_[i]] += integration_factor * (sigma_corr_grad_u_infinity * help);
      }
    } // end loop over quadrature points
    
    return integrals;
  }
  
  std::vector<Scalar> analyticSurfaceIntegrals()
  {
    AnalyticTriangle<Scalar> triangle(corners_, frontIntersectionIndices_);
    triangle.bind(dipole_position_, dipole_moment_);
    Vector surface_integrals_small = triangle.surfaceIntegral(frontFacingNormal_);
    
    std::vector<Scalar> surface_integrals(number_of_dofs, 0.0);
    for(size_t i = 0; i < 3; ++i) {
      surface_integrals[frontIntersectionIndices_[i]] = surface_integrals_small[i];
    }
    
    return surface_integrals;
  }
  
  std::vector<Scalar> analyticPatchIntegrals()
  {
    auto geometry = entity_.geometry();
    auto local_coords_dummy = referenceElement(geometry).position(0, 0);
    Tensor sigma_corr = sigma_;
    sigma_corr -= sigma_infinity_;
  
    // get gradients of local basis functions
    std::vector<Dune::FieldMatrix<Scalar, 1, dim>> basis_jacobians(number_of_dofs);
    Dune::FieldMatrix<Scalar, dim, number_of_dofs> lhs_matrix;
    fem_.localBasis().evaluateJacobian(local_coords_dummy, basis_jacobians);
    for(size_t i = 0; i < dim; ++i) {
      for(size_t j = 0; j < number_of_dofs; ++j) {
        lhs_matrix[i][j] = basis_jacobians[j][0][i];
      }
    }
      
    lhs_matrix.leftmultiply(geometry.jacobianInverseTransposed(local_coords_dummy));
    lhs_matrix.leftmultiply(sigma_corr);
    lhs_matrix *= 1.0 / (4.0 * Dune::StandardMathematicalConstants<Scalar>::pi() * sigma_infinity_[0][0]);
    
    Vector rhs(0.0);
    for(const auto& intersection : Dune::intersections(gridPtr_->leafGridView(), entity_)) {
      Vector outerNormal = intersection.centerUnitOuterNormal();
      duneuro::AnalyticTriangle<Scalar> triangle(intersection.geometry().corner(0), intersection.geometry().corner(1), intersection.geometry().corner(2));
      triangle.bind(dipole_position_, dipole_moment_);
      rhs += triangle.patchFactor() * outerNormal;
    }
    
    Dune::FieldVector<Scalar, number_of_dofs> integrals(0.0);
    lhs_matrix.umtv(rhs, integrals);
    
    std::vector<Scalar> integral_vector(number_of_dofs);
    for(size_t i = 0; i < number_of_dofs; ++i) {
      integral_vector[dof_to_vertex_indices_[i]] = integrals[i];
    }
    
    return integral_vector;
  }
  
  // compute integral sigma_infinity * u_infinity * (eta x (coil - x) / ||coil - x||^3) over triangle
  Vector numericSurfaceMagneticField(size_t integration_order)
  {
    // create infinity potential with source at dipole_position_
    UInfinity u_infinity(gridPtr_->leafGridView());
    Tensor sigma_infinity_inverse(sigma_infinity_);
    sigma_infinity_inverse.invert();
    u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    
    const auto& quad_rule = Dune::QuadratureRules<Scalar, dim - 1>::rule(intersection_.type(), integration_order);
    Vector magnetic_field(0.0);
    const auto& geometryWorld = intersection_.geometry();
    
    // perform numerical integration
    for(const auto quad_point : quad_rule) {
      Vector global_position = geometryWorld.global(quad_point.position());
      Scalar integration_factor = geometryWorld.integrationElement(quad_point.position()) * quad_point.weight();
      Scalar sigma_infinity_u_infinity;
      Dune::FieldVector<Scalar, 1> u_infinity_val;
      u_infinity.evaluateGlobal(global_position, u_infinity_val);
      sigma_infinity_u_infinity = sigma_infinity_[0][0] * u_infinity_val[0];
      
      Vector rhs = coil_position_ - global_position;
      Scalar norm_diff = rhs.two_norm();
      Scalar norm_cubed = norm_diff * norm_diff * norm_diff;
      rhs /= norm_cubed;
      Vector cross_product;
      Dune::PDELab::CrossProduct<dim, dim>(cross_product, -frontFacingNormal_, rhs);
      cross_product *= sigma_infinity_u_infinity * integration_factor;
      magnetic_field += cross_product;
    } // end loop over quadrature points
    
    return magnetic_field;
  }
  
  // compute integral sigma_corr_grad_u_infinity x ((coil - x) / ||coil - x||^3) over tetrahedron
  Vector numericPatchMagneticField(size_t integration_order)
  {
    // create gradient of infinity potential with source at dipole_position_
    UInfinityGradient grad_u_infinity(gridPtr_->leafGridView());
    Tensor sigma_infinity_inverse(sigma_infinity_);
    sigma_infinity_inverse.invert();
    grad_u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    
    Tensor sigma_corr = sigma_;
    sigma_corr -= sigma_infinity_;
    
    const auto& quad_rule = Dune::QuadratureRules<Scalar, dim>::rule(entity_.type(), integration_order);
    auto geometry = entity_.geometry();
    
    Vector magnetic_field(0.0);
    // perform numerical integration
    for(const auto& quad_point : quad_rule) {
      auto local_position = quad_point.position();
      Vector global_position = geometry.global(local_position);
      Scalar integration_factor = geometry.integrationElement(local_position) * quad_point.weight();
      
      Vector vec;
      grad_u_infinity.evaluateGlobal(global_position, vec);
      Vector sigma_corr_grad_u_infinity;
      sigma_corr.mv(vec, sigma_corr_grad_u_infinity);
      
      Vector rhs = coil_position_ - global_position;
      Scalar norm_diff = rhs.two_norm();
      Scalar norm_cubed = norm_diff * norm_diff * norm_diff;
      rhs /= norm_cubed;
      Vector cross_product;
      Dune::PDELab::CrossProduct<dim, dim>(cross_product, sigma_corr_grad_u_infinity, rhs);
      magnetic_field += cross_product * integration_factor;
    } // end loop over quadrature points
  
    return magnetic_field;
  }
  
  // compute integral (sigma grad(chi u_infinity)) x ((coil - x) / ||coil - x||^3) over the tetrahedron
  // we assume the front facing intersection to be contained in the patch, meaning that chi is 1 on nodes
  // 0, 1, 2 and 0 on node 3.
  Vector numericTransitionMagneticField(size_t integration_order)
  {
    // create u_infinity and grad_u_infinity with source at dipole_position_
    UInfinity u_infinity(gridPtr_->leafGridView());
    UInfinityGradient grad_u_infinity(gridPtr_->leafGridView());
    Tensor sigma_infinity_inverse(sigma_infinity_);
    sigma_infinity_inverse.invert();
    u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    grad_u_infinity.set_parameters(dipole_moment_, dipole_position_, sigma_infinity_, sigma_infinity_inverse);
    
    const auto& quad_rule = Dune::QuadratureRules<Scalar, dim>::rule(entity_.type(), integration_order);
    std::vector<Dune::FieldVector<Scalar, 1>> basis_vals;
    std::vector<Dune::FieldMatrix<Scalar, 1, dim>> basis_jacobians;
    auto geometry = entity_.geometry();
    Vector magnetic_field(0.0);
    
    // perform numerical integration
    for(const auto& quad_point : quad_rule) {
      auto local_position = quad_point.position();
      Vector global_position = geometry.global(local_position);
      Scalar integration_factor = geometry.integrationElement(local_position) * quad_point.weight();
      
      // compute lhs
      fem_.localBasis().evaluateJacobian(local_position, basis_jacobians);
      auto basis_jacobians_sum = basis_jacobians[0] + basis_jacobians[1] + basis_jacobians[2];
      Vector grad_chi;
      geometry.jacobianInverseTransposed(local_position).mv(basis_jacobians_sum[0], grad_chi);
      
      fem_.localBasis().evaluateFunction(local_position, basis_vals);
      Scalar chi = basis_vals[0][0] + basis_vals[1][0] + basis_vals[2][0];
      
      Dune::FieldVector<Scalar, 1> u_infinity_vec;
      u_infinity.evaluateGlobal(global_position, u_infinity_vec);
      Scalar u_infinity_val = u_infinity_vec[0];
      
      Vector grad_u_infinity_vec;
      grad_u_infinity.evaluateGlobal(global_position, grad_u_infinity_vec);
      
      Vector factor = u_infinity_val * grad_chi + chi * grad_u_infinity_vec;
      Vector lhs;
      sigma_.mv(factor, lhs);
      
      // compute rhs
      Vector rhs = coil_position_ - global_position;
      Scalar norm_diff = rhs.two_norm();
      Scalar norm_cubed = norm_diff * norm_diff * norm_diff;
      rhs /= norm_cubed;
      
      // compute cross product
      Vector cross_product;
      Dune::PDELab::CrossProduct<dim, dim>(cross_product, lhs, rhs);
      
      magnetic_field += cross_product * integration_factor;
    } // end loop over quadrature points
  
    return magnetic_field;
  }
  
  Scalar edgeLength() const
  {
    return edgeLength_;
  }
  
  const Entity& entity() 
  {
    return entity_;
  }
  
  const Intersection& frontFacingIntersection()
  {
    return intersection_;
  }
  
  private:
    Scalar edgeLength_;
    Tensor sigma_;
    Tensor sigma_infinity_;
    std::unique_ptr<Grid> gridPtr_;
    Entity entity_;
    Intersection intersection_;
    std::vector<int> dof_to_vertex_indices_;
    std::vector<int> vertex_to_dof_indices_;
    std::vector<int> frontIntersectionIndices_;
    FiniteElement fem_;
    std::vector<Vector> corners_;
    Vector frontFacingNormal_;
    Vector dipole_position_;
    Vector dipole_moment_;
    Vector coil_position_;
  }; // end definition of Mesh

} // end namespace duneuro
#endif // LOCAL_INTEGRATION_TEST_HH

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

#include <Python.h>
#include <dune/python/pybind11/numpy.h>
#include <dune/python/pybind11/operators.h>
#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>

namespace py = pybind11;
using Scalar = double;
enum {dim = 3};
using CoordinateType = Dune::FieldVector<Scalar, dim>;

// this is a smaller version of the corresponding code in duneuro-py
void register_coordinate_vector(py::module& m) {
  py::class_<CoordinateType>(m, 
                         "Coordinate", 
                         "3-dimensional Vector containing the cartesian coordinates of a point",
                         py::buffer_protocol())
    .def_buffer([] (CoordinateType& vector) -> py::buffer_info {
        return py::buffer_info(
          &vector[0],                                        // pointer to buffer
          sizeof(Scalar),                                    // size of one entry in bytes
          py::format_descriptor<Scalar>::format(),           // python struct for data type
          1,                                                 // number of dimensions
          {dim},                                             // dimension sizes
          {sizeof(Scalar)}                                   // stride sizes in bytes
        ); // end buffer_info constructor
      } // end lambda definition
    ) // end def_buffer
    .def(py::init<Scalar>(), "create Coordinate from scalar")
    .def(py::init(
      [](py::buffer buffer) {
        // check buffer format  
        py::buffer_info info = buffer.request();
        if(info.format != py::format_descriptor<Scalar>::format()) {
          DUNE_THROW(Dune::Exception, "buffer entries are of the wrong type");
        }
        if(info.ndim != 1) {
          DUNE_THROW(Dune::Exception, "buffer has to consist of 1 dimension, but consists of " << info.ndim << " dimensions");
        }
        if(info.shape[0] != dim) {
          DUNE_THROW(Dune::Exception, "buffer has to contain 3 entries, but contains " << info.shape[0] << " entries");
        }
        
        Scalar* data_ptr = static_cast<Scalar*>(info.ptr);
        return CoordinateType({data_ptr[0], data_ptr[1], data_ptr[2]});
      }) // end definition of lambda
      , "create Coordinate from Python buffer"
    ) // end definition of py::init
    .def(py::init(
      [](const py::list& value_list) {

        // validate list
        if(value_list.size() != dim) {
          DUNE_THROW(Dune::Exception, "list has to contain 3 entries, but contains" << value_list.size() << " entries");
        }
        // list validated
        
        // copy values to coordinate vector
        CoordinateType coordinate;
        std::transform(value_list.begin(), value_list.end(), coordinate.begin(), [] (const py::handle& handle) -> Scalar {return handle.cast<Scalar>();});
        return coordinate;
      }), // end definition of lambda
      "create coordinate vector from list"
    ) // end definition of py::init
    .def("__len__", [] (const CoordinateType& coordinate) {return coordinate.size();})
    .def("__getitem__",
      [](const CoordinateType& coordinate, size_t index) {
        return coordinate[index];
      } // end definition of lambda
    ) // end definition of __getitem__
    .def("__setitem__",
      [] (CoordinateType& coordinate, size_t index, Scalar value) {
        coordinate[index] = value;
      } // end definition of lambda
    ) // end definition of __setitem_
    .def("__str__",
      [](const CoordinateType& coordinate) {
        std::stringstream sstr;
        sstr << " Coordinate with entries [" << coordinate[0] << ", " << coordinate[1] << ", " << coordinate[2] << "]";
        return sstr.str();
      } // end definition of lambda
    ) // end definition of __str__
    // bind arithmetic operations
    .def(py::self += py::self)
    .def(py::self -= py::self)
    .def(py::self += Scalar())
    .def(py::self -= Scalar())
    .def(py::self *= Scalar())
    .def(py::self /= Scalar())
  ; // end definition of class
} // end register_coordinate_vector

void register_single_element_meshes(py::module m)
{
  // single tetrahedron mesh
  py::class_<duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>>(m, "SingleTetrahedronMesh", "class implementing the numerical and analytical integrals arising in the (localized) subtraction source model on a single tetrahedron and triangle")
    .def(py::init<Scalar, Scalar, Scalar>(), "create a single tetrahedron by specifying its edge length, the conductivity on the element and the conductivity around the dipole position",
                                            py::arg("edge_length") = 1.0, py::arg("conductivity_tetrahedron") = 1.79, py::arg("conductivity_dipole") = 0.33)
    .def("bindDipole", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::bindDipole, "bind dipole by specifying its position and moment")
    .def("bindCoil", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::bindCoil, "bind MEG coil by specifying its position")
    .def("numericSurfaceIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::numericSurfaceIntegrals, "numerically compute patch boundary integral for EEG forward problem",
         py::arg("integration_order"))
    .def("numericPatchIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::numericPatchIntegrals, "numerically compute patch integral for EEG forward problem",
         py::arg("integration_order"))
    .def("numericSurfaceMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::numericSurfaceMagneticField, "numerically compute patch boundary integral for MEG postprocessing",
         py::arg("integration_order"))
    .def("numericPatchMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::numericPatchMagneticField, "numerically compute patch integral for MEG postprocessing",
         py::arg("integration_order"))
    .def("numericTransitionMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::numericTransitionMagneticField, "numerically compute transition integral for MEG postprocessing",
         py::arg("integration_order"))
    .def("analyticSurfaceIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::analyticSurfaceIntegrals, "analytically compute patch boundary integral for EEG forward problem")
    .def("analyticPatchIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::tetrahedron>::analyticPatchIntegrals, "analytically compute patch integral for EEG forward problem")
    ; // end definition of class

  // single hexahedron mesh
  py::class_<duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>>(m, "SingleHexahedronMesh", "class implementing the numerical integrals arising in the (localized) subtraction source model on a single hexahedron and quadrilateral")
    .def(py::init<Scalar, Scalar, Scalar>(), "create a single hexahedron by specifying its edge length, the conductivity on the element and the conductivity around the dipole position",
                                            py::arg("edge_length") = 1.0, py::arg("conductivity_tetrahedron") = 1.79, py::arg("conductivity_dipole") = 0.33)
    .def("bindDipole", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::bindDipole, "bind dipole by specifying its position and moment")
    .def("bindCoil", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::bindCoil, "bind MEG coil by specifying its position")
    .def("numericSurfaceIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::numericSurfaceIntegrals, "numerically compute patch boundary integral for EEG forward problem",
         py::arg("integration_order"))
    .def("numericPatchIntegrals", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::numericPatchIntegrals, "numerically compute patch integral for EEG forward problem",
         py::arg("integration_order"))
    .def("numericSurfaceMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::numericSurfaceMagneticField, "numerically compute patch boundary integral for MEG postprocessing",
         py::arg("integration_order"))
    .def("numericPatchMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::numericPatchMagneticField, "numerically compute patch integral for MEG postprocessing",
         py::arg("integration_order"))
    .def("numericTransitionMagneticField", &duneuro::SingleElementMesh<duneuro::ElementType::hexahedron>::numericTransitionMagneticField, "numerically compute transition integral for MEG postprocessing",
         py::arg("integration_order"))
    ; // end definition of class
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
// Create bindings
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
PYBIND11_MODULE(singleElementPy, m) {
  register_coordinate_vector(m);
  py::implicitly_convertible<py::buffer, CoordinateType>();
  py::implicitly_convertible<py::list, CoordinateType>();
  register_single_element_meshes(m);
}

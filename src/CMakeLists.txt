add_library("singleTetrahedronPy" SHARED local_integration_test.cc)
target_link_libraries(singleTetrahedronPy ${PYTHON_LIBRARIES})
set_target_properties(singleTetrahedronPy PROPERTIES PREFIX "")

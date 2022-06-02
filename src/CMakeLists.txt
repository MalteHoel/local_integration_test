add_library("singleElementPy" SHARED local_integration_test.cc)
target_link_libraries(singleElementPy ${PYTHON_LIBRARIES})
set_target_properties(singleElementPy PROPERTIES PREFIX "")

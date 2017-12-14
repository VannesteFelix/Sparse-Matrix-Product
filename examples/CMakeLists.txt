set(EXAMPLES_TESTDATA
	testdata/matJ1eig.mtx
	testdata/matKeig.mtx)

foreach(f ${EXAMPLES_TESTDATA})
	configure_file(${f} "${CMAKE_CURRENT_BINARY_DIR}/${f}" COPYONLY)
endforeach()

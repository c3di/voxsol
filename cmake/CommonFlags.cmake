if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
	
	if (MSVC)
		add_compile_options("$<$<CONFIG:RELEASE>:/Ox>")
		add_compile_options("$<$<CONFIG:RELEASE>:/fp:fast>")
	endif()
endif(MSVC)


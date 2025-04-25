SET REPOSITORY_ROOT=%cd%\..\

:: Project include folder
SET GCCORE_INCLUDE=%REPOSITORY_ROOT%include

:: Api path
SET API_ROOT=%REPOSITORY_ROOT%api\

:: All APIs
SET SDL_ROOT=%API_ROOT%SDL2-2.0.5\
::SET SFML_ROOT=%API_ROOT%SFML-2.4.1\
SET GTEST_ROOT=%API_ROOT%gtest-1.8.0\vs2015\
SET GLEW_ROOT=%API_ROOT%glew-2.0.0\
SET LZ4_ROOT=%API_ROOT%lz4-1.7.5\vs2015\
SET GLM_ROOT=%API_ROOT%glm-0.9.9.0\

:: Set sm for cuda projects
SET CUDA_COMPUTE_CAPABILITY=compute_35,sm_35

:: Launch the solution
start GcCore.sln
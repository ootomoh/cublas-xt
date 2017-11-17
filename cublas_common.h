#pragma once

#ifndef __CUBLAS_XT__
#include <cublas.h>
#include <cublas_v2.h>
#else
#include <cublasXt.h>
#endif

#define CUBLAS_HANDLE_ERROR( err ) (cuBlasStatusError( err, __FILE__, __LINE__ ))
const char* cublasGetErrorString(cublasStatus_t status);
void cuBlasStatusError( cublasStatus_t err,const char *file,int line );

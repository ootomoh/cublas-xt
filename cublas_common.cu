#include <stdio.h>
#include <stdlib.h>
#include "cublas_common.h"

const char* cublasGetErrorString(cublasStatus_t status){
	switch(status){
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	default: return "unknown error";
	}
}
void cuBlasStatusError( cublasStatus_t err,const char *file,int line ) {
	if (err != CUBLAS_STATUS_SUCCESS) {
		printf( "cuBLAS Error:\n%s in %s at line %d\n", cublasGetErrorString(err),file, line );
		exit( EXIT_FAILURE );
	}
}


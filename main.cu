#include <iostream>
#include "cuda_common.h"
#include "cublas_common.h"
#include "matrix_array.h"

const int calc = 100;

#ifndef __CUBLAS_XT__
void gemm(cublasHandle_t cublas,mtk::MatrixXf& A,mtk::MatrixXf &B, mtk::MatrixXf &C){
	float alpha = 1.0f,beta = 0.0f;
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				A.getRows(),B.getCols(),A.getRows(),
				&alpha,
				A.getDevicePointer(),A.getRows(),
				B.getDevicePointer(),B.getRows(),
				&beta,
				C.getDevicePointer(),C.getRows()) );
}
#else
void gemm(cublasXtHandle_t cublas,mtk::MatrixXf& A,mtk::MatrixXf &B, mtk::MatrixXf &C){
	float alpha = 1.0f,beta = 0.0f;
	CUBLAS_HANDLE_ERROR( cublasXtSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				A.getRows(),B.getCols(),A.getRows(),
				&alpha,
				A.getDevicePointer(),A.getRows(),
				B.getDevicePointer(),B.getRows(),
				&beta,
				C.getDevicePointer(),C.getRows()) );
}
#endif

void testSgemm(int dim){
	std::cout<<dim<<" x "<<dim<<std::endl;
	mtk::MatrixXf A,B,C;
	A.setSize(dim,dim)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	B.setSize(dim,dim)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	C.setSize(dim,dim)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);

#ifndef __CUBLAS_XT__
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR( cublasCreate( &cublas ) );
#else
	cublasXtHandle_t cublas;
	CUBLAS_HANDLE_ERROR( cublasXtCreate( &cublas ));
	int devices[] = {0,1,2,3};
	CUBLAS_HANDLE_ERROR(cublasXtDeviceSelect(cublas,1,devices));
	CUBLAS_HANDLE_ERROR(cublasXtSetBlockDim(cublas,2) );
#endif
	cudaEvent_t start,stop;
	float elapsed_time;
	CUDA_HANDLE_ERROR(cudaEventCreate(&start));
	CUDA_HANDLE_ERROR(cudaEventCreate(&stop));
	// cache
	gemm(cublas,A,B,C);
	CUDA_HANDLE_ERROR( cudaEventRecord(start) );
	for(int c = 0;c < calc;c++){
		gemm(cublas,A,B,C);
	}
	CUDA_HANDLE_ERROR( cudaEventRecord( stop ) );
	CUDA_HANDLE_ERROR( cudaEventSynchronize( stop ));
	CUDA_HANDLE_ERROR( cudaEventElapsedTime( &elapsed_time, start, stop ));
	std::cout<<"cublasSgemm : "<<(elapsed_time/calc)<<" ms"<<std::endl;
	CUDA_HANDLE_ERROR( cudaEventDestroy( start ) );
	CUDA_HANDLE_ERROR( cudaEventDestroy( stop ) );
#ifndef __CUBLAS_XT__
	CUBLAS_HANDLE_ERROR( cublasDestroy( cublas ) );
#else
	CUBLAS_HANDLE_ERROR( cublasXtDestroy( cublas ));
#endif
}

int main(){
	for(int i=4;i<16;i++)
		testSgemm(1<<i);
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include <string.h>
#include <assert.h>	
#include "read_huge.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define M 10
#define D 3
#define N 10

// Definition of the kNN result struct
typedef struct knnresult{
  	int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  	double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  	int      m;       //!< Number of query points                 [scalar]
  	int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [m-by-d]
/*!

  \param  X      Query data points              [m-by-d]
  \param  Y      Corpus data points               [n-by-d]
  \param  m      Number of query points          [scalar]
  \param  n      Number of corpus points         [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

double k_select(int *arr, int n, int k);
void e_distance(double *X, double *Y, double *ED);
void f_distance(double *X, double *Y, double *ED);
void init_dist(double *X, double *Y, double *Dist);
void print_formated(double *X, int rows, int cols);
void calc_xsqr(double *X, double *Xsq);

int main(int argc, int *argv[]){
	int flag = 1;
    
	//allocate mem for X, Y and 
    double *X = (double*)malloc(M * D * sizeof(double*));
    double *Y = (double*)malloc(N * D * sizeof(double*));
	double *Dist = (double*)malloc(N * N * sizeof(double*));
	double *Xsq = (double*)malloc(M * sizeof(double*));
	double *Ysq = (double*)malloc(N * sizeof(double*));
	double *Dist_diagnostic = (double*)malloc(M * N * sizeof(double*));

	//TODO read from specific line (maybe implement using fseek())
	read_d(X, D);

	//copy X to Y (probably do it on first run only)
	//TODO block corpus Y so MxN doesn't explode
	if(flag){
		memcpy(Y, X, M*D*sizeof(double*));
	}

	
  	double start_time = MPI_Wtime();


	//printf("Original Data Point Set:");
	//print_formated(X, M, D);
	
	//calculate element-wise square of X and Y
	calc_xsqr(X, Xsq);
	calc_xsqr(Y, Ysq);

	//initiliase D 
	init_dist(Xsq, Ysq, Dist);
	
	//blas matrix-matrix calc -2 * X x (Y)T and adds it to D
	f_distance(X, Y, Dist);
	
	//performance metrics
	double end_time = MPI_Wtime();
  	double elapsed_time = end_time - start_time;
  	printf("Smart Elapsed time: %f seconds\n", elapsed_time);
	
	printf("Openblas results:");
	print_formated(Dist, M, N);

	start_time = MPI_Wtime();
	//diagnostics with triple loop dumb method 
	e_distance(X, Y, Dist_diagnostic);
	
	end_time = MPI_Wtime();
	elapsed_time = end_time - start_time;
	printf("Dumb Elapsed time: %f seconds \n", elapsed_time);

	free(X);
	free(Y);
	free(Dist);
}

void print_formated(double *X, int rows, int cols){
	printf("\n");
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			printf("%.4lf ", X[i*cols+j]);
		}
		printf("\n");
	}
	printf("\n");
}

void init_dist(double *Xsq, double *Ysq, double *Dist){
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Dist[i*N+j] = Xsq[i] + Ysq[j];
		}
	}
}

void calc_xsqr(double *X, double *Xsq){
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<D;j++){
			sum += X[i*D+j] * X[i*D+j];
		}
		Xsq[i] = sum;
	}
}

void f_distance(double *X, double *Y, double *Dist){
	//supposedly 1.0*X*Y -2.0*C -> C (if dimensionality resolved properly)
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
	//			M, N, D, 1.0, X, D, Y, D, 0.0, Dist, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				 M, N, D, -2.0, X, D, Y, D, 1.0, Dist, N);
}


//dumb version (maybe works but awful performance (?))
void e_distance(double *X, double *Y, double *Dist){
	for(int i=0; i<M; i++){
    	for(int j=0; j<N; j++) {
    		double sum = 0;
      		for (int k = 0; k < D; k++) {
        		sum += (X[i * D + k] - Y[j * D + k]) * (X[i * D + k] - Y[j * D + k]);
      		}
      	Dist[i * N + j] = sum;
    	}
	}
	
	printf("Diagnostics final results:\n");
	print_formated(Dist, M, N);
}


double k_select(int *arr, int n, int k)
{
    double pivot = arr[n / 2];  // choose pivot as middle element

    // partition the array around the pivot
    int i = 0, j = n - 1;
    while (i <= j)
    {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j)
        {
            double temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }

    // check if kth smallest element is in left subarray
    if (k <= j)
        return k_select(arr, j + 1, k);
    // check if kth smallest element is in right subarray
    else if (k > i)
        return k_select(arr + i, n - i, k - i);
    // kth smallest element is pivot
    else
        return pivot;
}

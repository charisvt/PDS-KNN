#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>
#include <string.h>
#include <assert.h>	
#include <stdint.h>
#include "read_huge.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))

size_t M = 1000;
size_t D = 3;
size_t N = 1000;
size_t k = 17;

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

double k_select(double *arr, int *nidx, int n, int k);
void e_distance(double *X, double *Y, double *ED);
void init_dist(double *X, double *Y, double *Dist);
void print_formated(double *X, int rows, int cols);
void calc_sqr(double *X, double *Xsq);
static inline void *MallocOrDie(size_t MemSize);

int main(int argc, int *argv[]){
	//mpi init
	int rank, num_procs=1;
    //char buf[256];
	MPI_Init(&argc, &argv);
	//id proc
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int flag = 1;
    
	//TODO BLOCK Y so that we avoid allocating big Dist
	//if X is split on a set number of mpi procs
	//then Mi can be really big so we have to keep Ni small
	//to keep Mi*Yi from exploding

	//allocate mem for X, Y and 
    double *X = (double*)MallocOrDie(M * D * sizeof(double*));
    double *Y = (double*)MallocOrDie(N * D * sizeof(double*));
	//double *Dist_diagnostic = (double*)MallocOrDie(M * N * sizeof(double*));

	//TODO read from specific line (maybe implement using fseek() )
	read_d(X, M, D, rank, num_procs);

	//copy X to Y (probably do it on first run only)
	//TODO block corpus Y so MxN doesn't explode
	if(flag){
		memcpy(Y, X, M*D*sizeof(double*));
	}

	
  	double start_time = MPI_Wtime();


	//printf("Original Data Point Set:");
	//print_formated(X, M, D);
	
	//performance metrics
	
	//start_time = MPI_Wtime();
	//diagnostics with triple loop dumb method 
	//e_distance(X, Y, Dist);
	//printf("Diagnostics final results:\n");
	//print_formated(Dist, M, N);

	//end_time = MPI_Wtime();
	//elapsed_time = end_time - start_time;
	//printf("Dumb Elapsed time: %f seconds \n", elapsed_time);

	knnresult r = kNN(X, Y, N, M, D, k);
	double end_time = MPI_Wtime();
	double elapsed_time = end_time - start_time;
	printf("Knn ended in %f seconds \n", elapsed_time);

	free(X);
	free(Y);
	//print_formated(r.ndist, M, k);
	free(r.ndist);
	free(r.nidx);
	//free(Dist_diagnostic);
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
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Dist[i*N+j] = Xsq[i] + Ysq[j];
		}
	}
}

void calc_sqr(double *X, double *Xsq){
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<D;j++){
			sum += X[i*D+j] * X[i*D+j];
		}
		Xsq[i] = sum;
	}
}

//dumb version (awful performance)
void e_distance(double *X, double *Y, double *Dist){
	#pragma omp parallel for
	for(int i=0; i<M; i++){
    	for(int j=0; j<N; j++) {
    		double sum = 0;
      		for (int k = 0; k < D; k++) {
        		sum += (X[i * D + k] - Y[j * D + k]) * (X[i * D + k] - Y[j * D + k]);
      		}
      	Dist[i * N + j] = sum;
    	}
	}
}

//TODO implement the whole kNN algo here
knnresult kNN(double * X, double * Y, int n, int m, int d, int k){
	double *Dist = (double*)MallocOrDie(N * N * sizeof(double*));
	double *Xsq = (double*)MallocOrDie(M * sizeof(double*));
	double *Ysq = (double*)MallocOrDie(N * sizeof(double*));
	double kth;
	knnresult r;
	r.m = m;
	r.k = k;
	r.nidx = (int*)MallocOrDie(m * k * sizeof(int*));
	r.ndist = (double*)MallocOrDie(m * k * sizeof(double*));
	int *ids = (int*)MallocOrDie(N * sizeof(int*));

	//init (global) indexes 
	//calculate element-wise square of X and Y
	calc_sqr(X, Xsq);
	calc_sqr(Y, Ysq);

	//initiliase D 
	init_dist(Xsq, Ysq, Dist);
	
	//blas matrix-matrix calc -2 * X x (Y)T and adds it to D
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				 M, N, D, -2.0, X, D, Y, D, 1.0, Dist, N);

	//TODO calc and popul knnresult r
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		//init indexes
		
		for(int j=0;j<N;j++){
			ids[j] = j;
		}

		//pointer arithmetics big brainz go brr	
		//k_select(Dist + i * N, ids, N, k); //TODO edit k_select to keep k-th order indeces too
		//memcpy(r.ndist + i * k, Dist + i, k * sizeof(double*)); //
		//memcpy(r.nidx + i * k, ids, k * sizeof(int*));

		free(ids);
	}


	free(Dist);
	free(Xsq);
	free(Ysq);
	return r;
}

double k_select(double *arr, int *nidx, int n, int k){
    double pivot = arr[n / 2];  // choose pivot as middle element
    int pivot_index = nidx[n / 2];

    // partition the array around the pivot
    int i = 0, j = n - 1;
    while(i <= j){
        while(arr[i] < pivot) i++;
        while(arr[j] > pivot) j--;
        if (i <= j){
            double temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            int temp_index = nidx[i];
            nidx[i] = nidx[j];
            nidx[j] = temp_index;
            i++;
            j--;
        }
    }

    // check if kth smallest element is in left subarray
    if (k <= j)
        return k_select(arr, nidx, j + 1, k);
    // check if kth smallest element is in right subarray
    else if (k > i)
        return k_select(arr + i, nidx + i, n - i, k - i);
    // kth smallest element is pivot
    else
        return pivot;
}


//this is a simple malloc wrapper that exits if a malloc fails
//source https://stackoverflow.com/questions/26831981/should-i-check-if-malloc-was-successful
static inline void *MallocOrDie(size_t MemSize){
    void *AllocMem = malloc(MemSize);
    if(!AllocMem && MemSize){
        printf("Could not allocate memory\n");
        exit(-1);
    }
    return AllocMem;
}
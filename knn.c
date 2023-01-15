#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include "useful.h"

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
void print_knnr(knnresult *r, int M, int k);

int main(int argc, char *argv[]){
	MPI_Init(&argc, &argv);
	int M = atoi(argv[1]), D = atoi(argv[2]), k = atoi(argv[3]);
    int N = M;

	//allocate mem for X, Y and 
    double *X = (double*)MallocOrDie(M * D * sizeof(double));
    double *Y = (double*)MallocOrDie(N * D * sizeof(double));

	read_d(X, M, D);

	memcpy(Y, X, M*D*sizeof(double));

	//performance metrics
	double start_time = MPI_Wtime();
	
	knnresult r = kNN(X, Y, N, M, D, k);
	
	double end_time = MPI_Wtime();
	double elapsed_time = end_time - start_time;
	
	print_knnr(&r, M, k);
	fprintf(stdout, "Knn ended in %f seconds \n", elapsed_time);
	
	//write to file 
	FILE *fp = fopen("knn_results.txt", "w");
	write_d(fp, r.ndist, r.nidx, M,  k);

	free(X);
	free(Y);
	free(r.ndist);
	free(r.nidx);
}


knnresult kNN(double * X, double * Y, int N, int M, int D, int k){
	double *Dist = (double*)MallocOrDie(N * N * sizeof(double));
	double *Xsq = (double*)MallocOrDie(M * sizeof(double));
	double *Ysq = (double*)MallocOrDie(N * sizeof(double));
	knnresult r;
	r.m = M;
	r.k = k;
	r.nidx = (int*)MallocOrDie(M * k * sizeof(int));
	r.ndist = (double*)MallocOrDie(M * k * sizeof(double));
	
 
	//calculate element-wise square of X and Y
	calc_sqr(X, Xsq, M, N, D);
	calc_sqr(Y, Ysq, M, N, D);

	//initiliase Dist
	init_dist(Xsq, Ysq, Dist, M, N);
	
	//blas matrix-matrix calc -2 * X x (Y)T and adds it to D
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				 M, N, D, -2.0, X, D, Y, D, 1.0, Dist, N);

	#pragma omp parallel for
	for(int i=0;i<M;i++){
		//allocate vector of size N to keep track of indexes as we swap distances in k_select
		int *ids = (int*)MallocOrDie(N * sizeof(int));
		for(int j=0;j<N;j++){
			ids[j] = j;
		}

		//k_select finds the kNN and we then copy them
		k_select(Dist + i * N, ids, N, k);
		memcpy(r.ndist + i * k, Dist + i * N, k * sizeof(double)); //
		memcpy(r.nidx + i * k, ids, k * sizeof(int));
		free(ids);
	}

	free(Dist);
	free(Xsq);
	free(Ysq);
	return r;
}


//k-select algorithm - k-th smallest element is at position k and every element <=k is partitioned to the left of k
//*! keep in mind that [0-(k-1)] elements are <=k but are not sorted !*
double k_select(double *arr, int *nidx, int n, int k){
    double pivot = arr[n / 2], temp_dist;  // choose pivot as middle element
    int temp_index;
    
	// partition the array around the pivot
    int i = 0, j = n - 1;
    while(i <= j){
        while(arr[i] < pivot) i++;
        while(arr[j] > pivot) j--;
        if (i <= j){
            temp_dist = arr[i];
            arr[i] = arr[j];
            arr[j] = temp_dist;
            temp_index = nidx[i];
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

void print_knnr(knnresult *r, int M, int k){
	//print the knns for the first 10 points
	fprintf(stdout, "KNN Results:\n");
	for(int i=0;i<M;i++){
		fprintf(stdout, "Global ID: %d\n", i);
		for(int j=0;j<k;j++){
			fprintf(stdout, "%d " , r->nidx[i * k + j]);
		}
		fprintf(stdout,"\n");
		for(int j=0;j<k;j++){
			printf("%.0lf ", r->ndist[i * k + j]);
		}
		fprintf(stdout, "\n\n");
	}
}
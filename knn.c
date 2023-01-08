#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>
#include <string.h>
#include <assert.h>	
#include <stdint.h>
#include <float.h>
#include "useful.h"
#include "knn_h.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define M 8000
#define D 3
#define N 8000
#define k 17


//knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

//double k_select(double *arr, int *nidx, int n, int k);
//void print_knnr(knnresult *r);
//void update_knnresult(knnresult *r, knnresult *new);

int main(int argc, char *argv[]){
	//mpi init
	int rank, world_size=1, first_run = 1;
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int next = (rank + 1) % world_size;
  	int prev = (rank - 1 + world_size) % world_size;

	//BlockID can be omitted since it's calculatable but use it (for now) for simplicity
	int BlockSize = M / world_size, BlockID = rank;

	//TODO BLOCK Y so that we avoid allocating big Dist
	//if X is split on a set number of mpi procs
	//then Mi can be really big so we have to keep Ni small
	//to keep Mi*Yi from exploding

	//allocate mem for X, Y and Z
    double *X = (double*)MallocOrDie(M * D * sizeof(double));
    double *Y = (double*)MallocOrDie(N * D * sizeof(double));
	double *Z = (double*)MallocOrDie(M * D * sizeof(double));

	//copy X to Y (first run only)
	//TODO block corpus Y so MxN doesn't explode
	if(first_run){
		BlockSize = read_d(X, M, D, rank, world_size);
		memcpy(Y, X, M*D*sizeof(double));
		//start listening for next points
	}else{
		//start listening for next points
		
	}

	//performance metrics
  	double start_time = MPI_Wtime();
	knnresult r = kNN(X, Y, N, M, D, k);
	double end_time = MPI_Wtime();
	double elapsed_time = end_time - start_time;
	print_knnr(&r);
	printf("Knn ended in %f seconds \n", elapsed_time);


	free(X);
	free(Y);
	free(r.ndist);
	free(r.nidx);
	//mpi final
	MPI_Finalize();
}
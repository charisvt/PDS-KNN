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

int main(int argc, char *argv[]){
	//mpi init
	int rank, world_size, first_run = 1, step = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("Running on %d nodes\n", world_size);
	int next = (rank + 1) % world_size;
  	int prev = (rank - 1 + world_size) % world_size;

	//BlockID can be omitted since it's calculatable but use it (for now) for simplicity
	int BlockSize, BlockID = rank;

	//TODO BLOCK Y so that we avoid allocating big Dist
	//if X is split on a set number of mpi procs
	//then Mi can be really big so we have to keep Ni small
	//to keep Mi*Yi from exploding

	//allocate mem for X, Y and Z
    double *X = (double*)MallocOrDie(M * D * sizeof(double));
    double *Y = (double*)MallocOrDie(N * D * sizeof(double));
	double *Z = (double*)MallocOrDie(M * D * sizeof(double));
	//try to avoid this temp (swap) pointer if possible
	double* temp = Y;
	knnresult r;

	//start timing
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	printf("%d ", world_size);
	//copy X to Y (first run only)
	//TODO block corpus Y so MxN doesn't explode
	while(step < world_size){
		if(first_run){
			BlockSize = read_d(X, M, D, rank, world_size);
			printf("%d", BlockSize);
			memcpy(Y, X, M*D*sizeof(double));

			MPI_Request send_request;
			MPI_Isend(Y, M * D, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
			
			r = kNN(X, Y, N, M, D, k);
			print_knnr(&r);
			//start receiving new points to do calcs on
			//MPI_Request rcv_request;
			//MPI_Irecv(Z, M * D, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &rcv_request);
		
			
			//we should send 

			//use this to wait for comms to complete before using z buffer
			//int recv_end;
			//while(!recv_end) MPI_Test(&request,&recv_end,MPI_STATUS_IGNORE);


			first_run = 0;
			step++;
		}else{
			
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//performance metrics
	double end_time = MPI_Wtime();
	double elapsed_time = end_time - start_time;
	//
	printf("Knn ended in %f seconds \n", elapsed_time);


	free(X);
	free(Y);
	free(Z);
	free(r.ndist);
	free(r.nidx);
	//mpi final
	MPI_Finalize();
}
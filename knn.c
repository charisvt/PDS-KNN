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

int M = 8000, D = 3, N = 8000, k = 17;

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
	int block_size = calc_blocksize(M, world_size), block_id = rank;

	//TODO BLOCK Y so that we avoid allocating big Dist
	//if X is split on a set number of mpi procs
	//then Mi can be really big so we have to keep Ni small
	//to keep Mi*Yi from exploding


	//allocate mem for X, Y and Z
    double *X = (double*)MallocOrDie(block_size * D * sizeof(double));
    double *Y = (double*)MallocOrDie(block_size * D * sizeof(double));
	double *Z = (double*)MallocOrDie(block_size * D * sizeof(double));
	//try to avoid this temp (swap) pointer if possible
	double* temp = Y;
	knnresult r, q;
	MPI_Request send_request, recv_request, send_status, recv_status;


	//start timing
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	printf("%d ", world_size);
	//copy X to Y (first run only)
	//TODO block corpus Y so MxN doesn't explode
	while(step < world_size){
		if(first_run){
			read_d(X, block_size, D, rank, world_size);
			memcpy(Y, X, block_size * D * sizeof(double));

			MPI_Isend(Y, block_size * D, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
			MPI_Irecv(Z, block_size * D, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request);

			r = kNN(X, Y, block_size, block_size, D, k, rank);

			MPI_Wait(&recv_request, &recv_status);
			MPI_Wait(&send_request, &send_status);

			//exchange pointers
			p_exchange(Y, Z);

			//use this to wait for comms to complete before using z buffer
			//int recv_end;
			//while(!recv_end) MPI_Test(&recv_request,&recv_end,MPI_STATUS_IGNORE);
			first_run = 0;
		}else{
			if(step < world_size - 1){ 
				MPI_Isend(Y, block_size * D, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
				MPI_Irecv(Z, block_size * D, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request);
			}

			//new results go into q
			q = kNN(X, Y, block_size, block_size, D, k, rank);
			update_knnresult(&r, &q);

			if(step < world_size - 1){
				MPI_Wait(&recv_request, &recv_status);
				MPI_Wait(&send_request, &send_status);
			}
		}
		step++;
	}
	if(rank == 0){
		FILE *fp = fopen("knn_results.txt", "a");
		if(fp == NULL){ 
			fprintf(stderr, "Error writing to file\n");
			exit(-1);
		}
		write_d(fp, r.ndist, r.nidx, block_size, k);
		for(int i=1; i < world_size ;i++){
			MPI_Recv(r.ndist, block_size * k, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(r.nidx, block_size * k, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			write_d(fp, r.ndist, r.nidx, block_size, k);
		}
	}else{
		//(buf, size, type, dest, tag, com_group)
		MPI_Send(r.ndist, block_size * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(r.nidx, block_size * k, MPI_INT, 0, 2, MPI_COMM_WORLD);
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
	free(q.ndist);
	free(q.nidx);
	//mpi final
	MPI_Finalize();
}

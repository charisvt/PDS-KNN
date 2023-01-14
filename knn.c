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


int main(int argc, char *argv[]){
	//mpi init
	int M = atoi(argv[1]), D = atoi(argv[2]), k = atoi(argv[3]);
	int rank, world_size, step = 0, lock = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Win win;
	MPI_Win_create(&lock, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0) fprintf(stdout, "Running on %d nodes\n", world_size);
	int next = (rank + 1) % world_size;
  	int prev = (rank - 1 + world_size) % world_size;

	int block_size = calc_blocksize(M, world_size), block_id = rank;

	//TODO BLOCK Y so that we avoid allocating big Dist
	//if X is split on a set number of mpi procs
	//then Mi can be really big so we have to keep Ni small
	//to keep Mi*Yi from exploding


	//allocate mem for X, Y and Z
    double *X = (double*)MallocOrDie(block_size * D * sizeof(double));
    double *Y = (double*)MallocOrDie(block_size * D * sizeof(double));
	double *Z = (double*)MallocOrDie(block_size * D * sizeof(double));
	knnresult r, q;
	r.k = k; r.m = block_size; q.k = k; q.m = block_size;
	r.ndist = (double*)MallocOrDie(block_size * k * sizeof(double));
	r.nidx = (int*)MallocOrDie(block_size * k * sizeof(int));
	q.ndist = (double*)MallocOrDie(block_size * k * sizeof(double));
	q.nidx = (int*)MallocOrDie(block_size * k * sizeof(int));

	MPI_Request send_request, recv_request; 
	MPI_Status send_status, recv_status;
	//fprintf(stdout, "Allocated main mem ok on node %d\n", rank);

	//start timing
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();
	
	//TODO block corpus Y so MxN doesn't explode
	while(step < world_size){
		if(step==0){
			//sequential reads with simple mutual exclusion just in case
			while(1){
				if(lock == 0) {
	        		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
					lock = 1;
					read_d(X, block_size, D, rank, world_size);
					lock = 0;
					MPI_Win_unlock(rank, win);
					break;
				}
			}
			// if(rank == 1) print_formated(X, block_size, D);
			memcpy(Y, X, block_size * D * sizeof(double));

			MPI_Isend(Y, block_size * D, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
			MPI_Irecv(Z, block_size * D, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request);

			kNN(X, Y, block_size, block_size, D, k, block_id, &r);
			// if(rank==1) print_knnr(&r);
			//debug region
			char filename[20];
			sprintf(filename, "debug-%d.txt", rank);
			FILE *dbg = fopen(filename, "w");
			write_d(dbg, r.ndist, r.nidx, block_size, k);
			fclose(dbg);
			//remove when done

			// int recv_end;
			// while(!recv_end) MPI_Test(&recv_request,&recv_end,MPI_STATUS_IGNORE);
			// int send_end;
			// while(!send_end) MPI_Test(&send_request,&send_end,MPI_STATUS_IGNORE);
			
			MPI_Wait(&recv_request, &recv_status);
			MPI_Wait(&send_request, &send_status);			
			//exchange pointers
			p_exchange(Y, Z);

			//use this to wait for comms to complete before using z buffer
			//int recv_end;
			//while(!recv_end) MPI_Test(&recv_request,&recv_end,MPI_STATUS_IGNORE);
		}else{
			if(step < world_size - 1){ 
				MPI_Isend(Y, block_size * D, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
				MPI_Irecv(Z, block_size * D, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request);
			}
			//new results go into q
			kNN(X, Y, block_size, block_size, D, k, block_id, &q);
			//this update function could be better
			update_knnresult(r.ndist, r.nidx, q.ndist, q.nidx, block_size, k);
			fprintf(stdout, "Unique debug %d\n", rank);
			if(step < world_size - 1){
				MPI_Wait(&recv_request, &recv_status);
				MPI_Wait(&send_request, &send_status);
			}
		}
		block_id = (block_id + 1) % world_size;
		step++;
	}

	//gather results
	if(rank == 0){
		FILE *fp = fopen("knn_results.txt", "w");
		if(fp == NULL){ 
			fprintf(stderr, "Error writing to file\n");
			exit(-1);
		}
		write_d(fp, r.ndist, r.nidx, block_size, k);
		fclose(fp);
		fp = fopen("knn_results.txt", "a");
		for(int i=1; i < world_size ;i++){
			MPI_Recv(r.ndist, block_size * k, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(r.nidx, block_size * k, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			write_d(fp, r.ndist, r.nidx, block_size, k);
		}
		fclose(fp);
	}else{
		//(buf, size, type, dest, tag, com_group)
		MPI_Send(r.ndist, block_size * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(r.nidx, block_size * k, MPI_INT, 0, 2, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	//performance metrics
	if(rank == 0){
		double end_time = MPI_Wtime();
		double elapsed_time = end_time - start_time;
		fprintf(stdout, "Knn ended in %f seconds \n", elapsed_time);
	}
	
	free(X);
	free(Y);
	free(Z);
	free(r.nidx);
    free(r.ndist);
	free(q.ndist);
	free(q.nidx);
	//mpi final
	MPI_Win_free(&win);
	MPI_Finalize();
}

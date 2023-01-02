#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

void main(int argc, char *argv[]){
    int rank, num_procs;
    char buf[256];
    //initialise
    MPI_Init(&argc, &argv);
    
    //id proc
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //total procs
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //main proc has id 0
    if(rank==0){
        printf("Running %d procs\n", num_procs);
        //message every other proc
        for(int i=1; i<num_procs;i++){
            //init buffer
            sprintf(buf, "Hello %d\n", i);
            //send by passing (buffer, buffer_len, buffer type, proc id, msg code*, comm group)  
            MPI_Send(buf, 256, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        //recv from all other procs
        for(int i=1; i<num_procs;i++){
            //recv by specing (buffer, buffer_len, buffer type, proc id, msg code*, comm group, comm status)
            MPI_Recv(buf, 256, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", buf);
        }
    }
    else{
        //recv from main (id 0)
        MPI_Recv(buf, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //test
        assert(memcmp(buf, "Hello ", 6) == 0);
        //msg to main (id 0)
        sprintf(buf, "Proccess %d reporting for duty\n", rank);
        MPI_Send(buf, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    //finalise
    MPI_Finalize();
}
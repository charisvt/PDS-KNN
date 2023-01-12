int read_d(double *X, int total_lines, int dim, int rank, int num_procs){
    //open a file pointer
	FILE *fp = fopen("knn_dataset.txt", "r");
    if(fp == NULL){
      exit(-1);
    }
    
	//CHECK THAT RANGES ARE CORECT - LAST MPI_PROC SHOULD READ TO EOF
	//Start reading file
    char line[256];
    int line_num = 0, block_size = total_lines / num_procs, low_bound, up_bound ;

	//check if we run on a single proc
	if(num_procs == 1){
		up_bound = total_lines;
	}else{
		//if we can't split total lines perfectly into num_proc blocks
		//allocate bigger block and have last proc get any remainding lines
		if(total_lines % num_procs != 0){
			block_size = total_lines / (num_procs -1);
		}
    	low_bound = block_size*rank;
		up_bound = block_size*(rank+1);
	}
	
	
    while(fgets(line, sizeof(line), fp)){
      	//only read lines that correspond to the mpi process 
		if(line_num>=low_bound && line_num<up_bound){
			char* tokens[dim];
      		line[strcspn(line, "\n")] = '\0';
      		char* token = strtok(line, " ");
      		int i = 0;
      		while (token != NULL) {
        		tokens[i] = token;
				X[line_num * dim + i] = atof(token);
				i++;
        		token = strtok(NULL, " ");
      		}
		}
      	line_num++;
    }
	//the last proc fills with points at infinity
	if(rank == num_procs-1 && (total_lines % num_procs)!=0 && num_procs != 1){
        int remaining = block_size - (total_lines%num_procs);
        for(int i = 0; i < remaining*dim; i++){
            X[line_num*dim+i] = DBL_MAX;
        }
    }
	//returns how many lines were read
	return line_num;
}


void print_formated(double *X, int rows, int cols){
	printf("\n");
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			printf("%.0lf ", X[i*cols+j]);
		}
		printf("\n");
	}
	printf("\n");
}

//dumb version (awful performance)
void e_distance(double *X, double *Y, double *Dist, int M, int N, int D){
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
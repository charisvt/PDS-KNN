void read_d(double *X, int total_lines, int dim, int rank, int num_procs){
    //open a file pointer
	FILE *fp = fopen("knn_dataset.txt", "r");
    if(fp == NULL){
      perror("File not found");
    }
    
	//TODO maybe find a way to calculate a proper offset
	//and use fseek() to point to the corresponding lines 
	//directly instead of skipping through every block of lines
	
	//CHECK THAT RANGES ARE CORECT - LAST MPI_PROC SHOULD READ TO EOF
	//Start reading file
    char line[256];
    int line_num = 0;
    int low_bound = (total_lines / num_procs)*rank;
	int up_bound = (total_lines / num_procs)*(rank+1);
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
}
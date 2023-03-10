int calc_blocksize(int total_lines, int num_procs){
	if(total_lines % num_procs != 0) return total_lines / (num_procs -1);
	return total_lines / num_procs;
}

int read_d(double *X, int block_size, int dim, int rank, int num_procs){
	FILE *fp = fopen("knn_dataset.txt", "r");
    if(fp == NULL){
		fprintf(stdout, "Error: Could not open file\n");
		exit(-1);
    }
    
    char line[256];
    int line_num = 0, low_bound = 0, up_bound;

	low_bound = block_size*rank;
	up_bound = block_size*(rank+1);
	
	if(rank==1){
		fprintf(stdout, "read got called with %d\n", block_size);
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
	int remaining = line_num % block_size;
    for(int i = 0; i < remaining*dim; i++){
		X[line_num*dim+i] = 5.1e20;
    }
	fclose(fp);
	return line_num;
}

void write_d(FILE *fp, double *dist, int *indeces, int m, int kn){
	for(int i=0;i<m;i++){
		for(int j=0;j<kn;j++) fprintf(fp, "%lf ", dist[i * kn + j]);
		for(int j=0;j<kn;j++) fprintf(fp, "%d ", indeces[i * kn + j]);
		fprintf(fp, "\n");
	}
}

//pointer swap
static inline void p_exchange(double *y, double *z){
	double *temp = y;
	y = z;
	z = temp;
}

void print_formated(double *X, int rows, int cols){
	fprintf(stdout, "\n");
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			fprintf(stdout, "%.0lf ", X[i*cols+j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
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
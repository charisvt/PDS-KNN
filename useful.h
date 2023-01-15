void read_d(double *X, int total_lines, int dim){
    //open a file pointer
	FILE *fp = fopen("knn_dataset.txt", "r");
    if(fp == NULL){
      perror("File not found");
    }

	//Start reading file
    char line[256];
    int line_num = 0;

    while(fgets(line, sizeof(line), fp)){
      	
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
      	line_num++;
    }
}

//this is a simple malloc wrapper that exits if a malloc fails
//source https://stackoverflow.com/questions/26831981/should-i-check-if-malloc-was-successful
static inline void *MallocOrDie(int MemSize){
    void *AllocMem = malloc(MemSize);
    if(!AllocMem && MemSize){
        fprintf(stdout, "Could not allocate memory\n");
        exit(-1);
    }
    return AllocMem;
}

void init_dist(double *Xsq, double *Ysq, double *Dist, int M, int N){
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Dist[i*N+j] = Xsq[i] + Ysq[j];
		}
	}
}

void calc_sqr(double *X, double *Xsq, int M, int N, int D){
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<D;j++){
			sum += X[i*D+j] * X[i*D+j];
		}
		Xsq[i] = sum;
	}
}

void write_d(FILE *fp, double *dist, int *indeces, int m, int kn){
	for(int i=0;i<m;i++){
		for(int j=0;j<kn;j++) fprintf(fp, "%lf ", dist[i * kn + j]);
		for(int j=0;j<kn;j++) fprintf(fp, "%d ", indeces[i * kn + j]);
		fprintf(fp, "\n");
	}
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
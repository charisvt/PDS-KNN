void read_d(double *X, int dim){
    //open a file pointer
	FILE *fp = fopen("test.txt", "r");
    if(fp == NULL){
      perror("File not found");
    }
    
	//Start reading file
    char line[256];
    int line_num = 0;
    
    while(fgets(line, sizeof(line),fp)){
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
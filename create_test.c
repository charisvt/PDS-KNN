#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, int **argv){
    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    if(argc<3){
        printf("pass args dummy :)");
        return 0;
    }
    srand(time(NULL));
    FILE *fp = fopen("test.txt", "w");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            double x = (double)rand() / RAND_MAX;
            fprintf(fp, "%.2f ", x);
        }
        fprintf(fp, "\n");
    }

// Close the file
fclose(fp);

return 0;
}
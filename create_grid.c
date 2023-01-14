#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_STEP 1

int main(int argc, char** argv) {
    // Open a file for writing the dataset
    int MIN_COORDINATE = atoi(argv[1]), MAX_COORDINATE = atoi(argv[2]), NUM_FEATURES = atoi(argv[3]) ;
    FILE* fp = fopen("knn_dataset.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // Generate points on a regular grid
    for (double i = MIN_COORDINATE; i <= MAX_COORDINATE; i += GRID_STEP) {
        for (double j = MIN_COORDINATE; j <= MAX_COORDINATE; j += GRID_STEP) {
            for(double k = MIN_COORDINATE; k <= MAX_COORDINATE; k += GRID_STEP){
                fprintf(fp, "%lf %lf %lf\n", i, j, k);
            }
        }
    }

    // Close the file
    fclose(fp);
    return 0;
}

# Sequential all-kNN V.0
Usage:
Compile with:
mpi knn.c -o knn -fopenmp -lopenblas -O3
Run with:
./knn {num of points M} {num of dimensions D} {num of neighbors k}
## Parallel all-kNN
# Usage:
# Compile with:
mpi knn.c -o knn -fopenmp -lopenblas -O3
# Run with:
mpiexec -n {num of nodes P} ./knn {num of points M} {num of dimensions D} {num of neighbors k}

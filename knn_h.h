// Definition of the kNN result struct
typedef struct knnresult{
  	int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  	double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  	int      m;       //!< Number of query points                 [scalar]
  	int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [m-by-d]
/*!

  \param  X      Query data points              [m-by-d]
  \param  Y      Corpus data points               [n-by-d]
  \param  m      Number of query points          [scalar]
  \param  n      Number of corpus points         [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/

//this is a simple malloc wrapper that exits if a malloc fails
//source https://stackoverflow.com/questions/26831981/should-i-check-if-malloc-was-successful
static inline void *MallocOrDie(size_t MemSize){
    void *AllocMem = malloc(MemSize);
    if(!AllocMem && MemSize){
        printf("Could not allocate memory\n");
        exit(-1);
    }
    return AllocMem;
}

void calc_sqr(double *X, double *Xsq, int M, int D);
void init_dist(double *Xsq, double *Ysq, double *Dist, int M, int N);
double k_select(double *arr, int *nidx, int n, int k);

//kNN algo here
knnresult kNN(double * X, double * Y, int M, int N, int D, int k){
	double *Dist = (double*)MallocOrDie(N * N * sizeof(double));
	double *Xsq = (double*)MallocOrDie(M * sizeof(double));
	double *Ysq = (double*)MallocOrDie(N * sizeof(double));
	double kth;
	knnresult r;
	r.m = M;
	r.k = k;
	r.nidx = (int*)MallocOrDie(M * k * sizeof(int));
	r.ndist = (double*)MallocOrDie(M * k * sizeof(double));
	//allocate vector of size N to keep track of indexes as we swap distances in k_select
	
 
	//calculate element-wise square of X and Y
	calc_sqr(X, Xsq, M, D);
	calc_sqr(Y, Ysq, N, D);

	//initiliase Dist
	init_dist(Xsq, Ysq, Dist, M, N);
	
	//blas matrix-matrix calc -2 * X x (Y)T and adds it to D
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				 M, N, D, -2.0, X, D, Y, D, 1.0, Dist, N);

	//print_formated(Dist, M, N);
	//TODO calc and populate knnresult r
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		//TODO init indexes according to global indexes
		//should be sth like g_idx = BlockID * BlockSize + j  
		//this allocation is only needed cause of thread parallelism
		//maybe its too costly and we need to do it in r.nidx
		int *ids = (int*)MallocOrDie(N * sizeof(int));
		for(int j=0;j<N;j++){
			ids[j] = j;
		}

		//pointer arithmetics big brainz go brr	
		k_select(Dist + i * N, ids, N, k);
		memcpy(r.ndist + i * k, Dist + i * N, k * sizeof(double)); //
		memcpy(r.nidx + i * k, ids, k * sizeof(int));
		free(ids);
	}

	free(Dist);
	free(Xsq);
	free(Ysq);
	return r;
}


//k-select algorithm - k-th smallest element is at position k and every element <=k is partitioned to the left of k
//*! keep in mind that [0-(k-1)] elements are <=k but are not sorted !*
double k_select(double *arr, int *nidx, int n, int k){
    double pivot = arr[n / 2], temp_dist;  // choose pivot as middle element
    int pivot_index = nidx[n / 2], temp_index;
    
	// partition the array around the pivot
    int i = 0, j = n - 1;
    while(i <= j){
        while(arr[i] < pivot) i++;
        while(arr[j] > pivot) j--;
        if (i <= j){
            temp_dist = arr[i];
            arr[i] = arr[j];
            arr[j] = temp_dist;
            temp_index = nidx[i];
            nidx[i] = nidx[j];
            nidx[j] = temp_index;
            i++;
            j--;
        }
    }

    // check if kth smallest element is in left subarray
    if (k <= j)
        return k_select(arr, nidx, j + 1, k);
    // check if kth smallest element is in right subarray
    else if (k > i)
        return k_select(arr + i, nidx + i, n - i, k - i);
    // kth smallest element is pivot
    else
        return pivot;
}


void print_knnr(knnresult *r){
	//print the knns for the first 10 points
	printf("KNN Results:\n");
	for(int i=0;i<r->m;i++){
		printf("Global ID: %d\n", i);
		for(int j=0;j<r->k;j++){
			printf("%d " , r->nidx[i * r->k + j]);
		}
		printf("\n");
		for(int j=0;j<r->k;j++){
			printf("%.0lf ", r->ndist[i * r->k + j]);
		}
		printf("\n\n");
	}
}

void update_knnresult(knnresult *r, knnresult *new){
	for (int i = 0; i < r->m; i++){
		for (int j = 0; j < r->k; j++){
			int idx = i * r->k + j;  // index of current element in ndist and nidx arrays
			if (new->ndist[idx] < r->ndist[idx]){  // update ndist and nidx if new is smaller
				r->ndist[idx] = new->ndist[idx];
				r->nidx[idx] = new->nidx[idx];
			}
		}
	}
}


void init_dist(double *Xsq, double *Ysq, double *Dist, int M, int N){
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Dist[i*N+j] = Xsq[i] + Ysq[j];
		}
	}
}

void calc_sqr(double *X, double *Xsq, int M, int D){
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<D;j++){
			sum += X[i*D+j] * X[i*D+j];
		}
		Xsq[i] = sum;
	}
}
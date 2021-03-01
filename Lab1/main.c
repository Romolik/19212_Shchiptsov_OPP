#include <mpi.h> // Подключение библиотеки MPI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

struct timeval tv1, tv2;

void mulMatrix (double *A, double *x, double *condition,
				const size_t N, const size_t M, size_t rank) {

	for (size_t i = 0; i < M; ++i) {
		double cur = 0;
		for (size_t j = 0; j < N; ++j) {
			cur += (A[i * N + j] * x[rank * M + j]);
		}
		condition[rank * M + i] = cur;
	}
}

void difCol (double *Ax, double *b, const size_t M, size_t rank) {
	for (size_t i = 0; i < M; ++i) {
		Ax[i] -= b[rank * M + i];
	}
}

double normCounting (double *Ax, const size_t M) {
	double res = 0;
	for (size_t i = 0; i < M; ++i) {
		res += (Ax[i] * Ax[i]);
	}
	double norm;
	MPI_Allreduce(&res, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return sqrt (norm);
}

double normCountingForB (double *b, const size_t M, size_t rank, size_t N) {
	double res = 0;
	for (size_t i = rank * M; i < (rank * M + M * N); ++i) {
		res += (b[i] * b[i]);
	}
	double norm;
	MPI_Allreduce(&res, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return sqrt (norm);
}

void mulScalarCol (double *Ax, const double scalar,
				   const size_t M) {
	for (size_t i = 0; i < M; ++i) {
		Ax[i] *= scalar;
	}
}

void printCol (double *x, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		printf ("%f\n", x[i]);
	}
}

int criterionCompletion (double *Ax, double *b, const size_t M, size_t rank,
						 const size_t N, const double EPSILON) {
	difCol (Ax, b, M, rank);
	double firsrNorm = normCounting (Ax, M);
	double secondNorm = normCountingForB( b, M, rank, N);
	return (firsrNorm / secondNorm) >= EPSILON;
}

void simpleIterationMethod (double *A, double *b, double *x, const size_t N,
							const size_t M, size_t rank, const double EPSILON) {
	const double t = 0.0001;
	double *condition = NULL;
	condition = (double *)malloc (N * sizeof (double));
	double *Ax = NULL;
	Ax = (double *)malloc (M * sizeof (double));

	if (condition == NULL || Ax == NULL) {
		free (Ax);
		free (condition);
		printf ("memory allocation error");
		return;
	}

	mulMatrix (A, x, condition, N, M, rank);

	while (criterionCompletion (condition, b, M, rank, N, EPSILON)) {
		mulMatrix (A, x, Ax, N, M, rank);
		difCol (Ax, b, M, rank);
		mulScalarCol (Ax, t, M);
		difCol (x, Ax, M, rank);
		mulMatrix (A, x, condition, N, M, rank);
	}

	free (condition);
	free (Ax);
}

int main (int argc, char *argv[]) {
	int size, rank;
	MPI_Init (&argc, &argv);               // Инициализация MPI
	MPI_Comm_size (MPI_COMM_WORLD, &size); // Получение числа процессов
	MPI_Comm_rank (MPI_COMM_WORLD, &rank); // Получение номера процесса
	const size_t N = 8;
	const size_t M = N / size;
	const double EPSILON = 0.00001;
	double *A = NULL;
	A = (double *)malloc (N * M * sizeof (double));
	double *b = NULL;
	b = (double *)malloc (N * sizeof (double));
	double *x = NULL;
	x = (double *)malloc (N * sizeof (double));

	if (x == NULL || b == NULL || A == NULL) {
		free (A);
		free (b);
		free (x);
		printf ("memory allocation error");
		MPI_Finalize (); // Завершение работы MPI
		return 0;
	}

	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			if ((rank * M + i) == j) {
				A[i * N + j] = 2;
			} else {
				A[i * N + j] = 1;
			}
		}
	}

	for (size_t i = 0; i < N; ++i) {
		x[i] = 0;
		b[i] = N + 1;
	}

	gettimeofday(&tv1, NULL);
	simpleIterationMethod (A, b, x, N, M, rank, EPSILON);
	gettimeofday(&tv2, NULL);
	if (rank == 0) {
		printCol (x, N);
	}
	double dt_sec = (tv2.tv_sec - tv1.tv_sec);
	double dt_usec = (tv2.tv_usec - tv1.tv_usec);
	double dt = dt_sec + 1e-6*dt_usec;
	double maxdt;
	MPI_Allreduce(&dt, &maxdt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("\nTime diff %lf\n", maxdt);
	}

	free (A);
	free (b);
	free (x);
	MPI_Finalize (); // Завершение работы MPI
	return 0;
}
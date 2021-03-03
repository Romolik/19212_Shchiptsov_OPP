#include <mpi.h> // Подключение библиотеки MPI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

struct timeval tv1, tv2;

void printCol (double *x, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		printf ("%f\n", x[i]);
	}
}

void mulMatrix (double *A, double *x, double *condition,
				const size_t N, const size_t M) {
	double* allX = (double*)malloc(N * sizeof(double));
	if (allX == NULL) {
		printf ("Error allocation memory!");
		return;
	}
	MPI_Allgather(x, M, MPI_DOUBLE, allX, M, MPI_DOUBLE, MPI_COMM_WORLD);
	for (size_t i = 0; i < M; ++i) {
		double cur = 0;
		for (size_t j = 0; j < N; ++j) {
			cur += (A[i * N + j] * allX[j]);
		}
		condition[i] = cur;
	}
	free (allX);
}

void difCol (double *Ax, double *b, const size_t M) {
	for (size_t i = 0; i < M; ++i) {
		Ax[i] -= b[i];
	}
}

double normCounting (double *Ax, const size_t M) {
	double res = 0;
	for (size_t i = 0; i < M; ++i) {
		res += (Ax[i] * Ax[i]);
	}
	double norm;
	MPI_Allreduce (&res, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return sqrt (norm);
}

void mulScalarCol (double *Ax, const double scalar,
				   const size_t M) {
	for (size_t i = 0; i < M; ++i) {
		Ax[i] *= scalar;
	}
}

int criterionCompletion (double *Ax, double *b, const size_t M,
						 const double EPSILON) {
	difCol (Ax, b, M);
	double firsrNorm = normCounting (Ax, M);
	double secondNorm = normCounting (b, M);
	return (firsrNorm / secondNorm) >= EPSILON;
}

void simpleIterationMethod (double *A, double *b, double *x, const size_t N,
							const size_t M, size_t rank, size_t size, const
							double EPSILON) {
	const double t = 0.00001;
	double *condition = NULL;
	condition = (double *)malloc (M * sizeof (double));
	double *Ax = NULL;
	Ax = (double *)malloc (M * sizeof (double));

	if (condition == NULL || Ax == NULL) {
		free (Ax);
		free (condition);
		printf ("memory allocation error");
		return;
	}

	mulMatrix (A, x, condition, N, M);
	int criretion = criterionCompletion (condition, b, M, EPSILON);
	while (criretion) {
		mulMatrix (A, x, Ax, N, M);
		difCol (Ax, b, M);
		mulScalarCol (Ax, t, M);
		difCol (x, Ax, M);
		mulMatrix (A, x, condition, N, M);
		criretion = criterionCompletion (condition, b, M, EPSILON);
	}

	free (condition);
	free (Ax);
}

int main (int argc, char *argv[]) {
	int size, rank;
	MPI_Init (&argc, &argv);               // Инициализация MPI
	MPI_Comm_size (MPI_COMM_WORLD, &size); // Получение числа процессов
	MPI_Comm_rank (MPI_COMM_WORLD, &rank); // Получение номера процесса
	const size_t N = 1024;
	const size_t M = N / size;
	const double EPSILON = 0.00001;
	double *A = NULL;
	A = (double *)malloc (N * M * sizeof (double));
	double *b = NULL;
	b = (double *)malloc (M * sizeof (double));
	double *x = NULL;
	x = (double *)malloc (M * sizeof (double));

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
		x[i] = 0;
		b[i] = N + 1;
	}

	gettimeofday (&tv1, NULL);
	simpleIterationMethod (A, b, x, N, M, rank, size, EPSILON);
	gettimeofday (&tv2, NULL);

	if (rank == 1) {
		printCol (x, M);
	}

	double dt_sec = (tv2.tv_sec - tv1.tv_sec);
	double dt_usec = (tv2.tv_usec - tv1.tv_usec);
	double dt = dt_sec + 1e-6 * dt_usec;
	double maxdt;

	MPI_Allreduce (&dt, &maxdt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf ("\nTime diff %lf\n", maxdt);
	}

	free (A);
	free (b);
	free (x);
	MPI_Finalize (); // Завершение работы MPI
	return 0;
}
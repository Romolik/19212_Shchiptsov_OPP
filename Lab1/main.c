#include <mpi.h> // Подключение библиотеки MPI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void mulMatrix (double *A, double *x, double *condition, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		double cur = 0;
		for (size_t j = 0; j < N; ++j) {
			cur += (A[i * N + j] * x[j]);
		}
		condition[i] = cur;
	}
}

void difCol (double *Ax, double *b, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		Ax[i] -= b[i];
	}
}

double normCounting (double *Ax, const size_t N) {
	double res = 0;
	for (size_t i = 0; i < N; ++i) {
		res += (Ax[i] * Ax[i]);
	}
	return sqrt (res);
}

void mulScalarCol (double *Ax, const double scalar, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		Ax[i] *= scalar;
	}
}

void printMatrix (double *A, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			printf ("%f\t", A[i * N + j]);
		}
		printf ("\n");
	}
}

void printCol (double *Ax, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		printf ("%f\n", Ax[i]);
	}
}

int criterionCompletion (
	double *Ax, double *b, const size_t N,
	const double EPSILON) {
	difCol (Ax, b, N);
	double firsrNorm = normCounting (Ax, N);
	double secondNorm = normCounting (b, N);
	return (firsrNorm / secondNorm) >= EPSILON;
}

void simpleIterationMethod (
	double *A, double *b, double *x,
	const size_t N, const double EPSILON) {
	const double t = 0.0001;
	double *condition = NULL;
	condition = (double *) malloc (N * sizeof (double));
	double *Ax = NULL;
	Ax = (double *) malloc (N * sizeof (double));

	if (condition == NULL || Ax == NULL) {
		free (Ax);
		free (condition);
		printf ("memory allocation error");
		return;
	}

	mulMatrix (A, x, condition, N);

	while (criterionCompletion (condition, b, N, EPSILON)) {
		mulMatrix (A, x, Ax, N);
		difCol (Ax, b, N);
		mulScalarCol (Ax, t, N);
		difCol (x, Ax, N);
		mulMatrix (A, x, condition, N);
	}

	free (condition);
	free (Ax);
	printCol (x, N);
}

int main (int argc, char *argv[]) {
	int size, rank;
	MPI_Init(&argc, &argv);               // Инициализация MPI
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение числа процессов
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение номера процесса
	const size_t N = 8;
	const size_t M = N / size;
	const double EPSILON = 0.00001;
	double *A = NULL;
	A = (double *) malloc (N * M * sizeof (double));
	double *b = NULL;
	b = (double *) malloc (N * sizeof (double));
	double *x = NULL;
	x = (double *) malloc (N * sizeof (double));

	if (x == NULL || b == NULL || A == NULL) {
		free (A);
		free (b);
		free (x);
		printf ("memory allocation error");
		MPI_Finalize(); // Завершение работы MPI
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

	simpleIterationMethod (A, b, x, N, EPSILON);

	free (A);
	free (b);
	free (x);
	MPI_Finalize(); // Завершение работы MPI
	return 0;
}
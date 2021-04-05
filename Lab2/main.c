#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

void printCol (double *x, const size_t N) {
	for (size_t i = 0; i < N; ++i) {
		printf ("%f\n", x[i]);
	}
}

void mulMatrix (double *A, double *x, double *condition,
				const size_t N) {

	#pragma omp parallel for
	for (size_t i = 0; i < N; ++i) {
		double cur = 0;
		for (size_t j = 0; j < N; ++j) {
			cur += (A[i * N + j] * x[j]);
		}
		condition[i] = cur;
	}
}

void difCol (double *Ax, double *b, const size_t N) {
	#pragma omp parallel for
	for (size_t i = 0; i < N; ++i) {
		Ax[i] -= b[i];
	}
}

double countNorm (double *Ax, const size_t N) {
	double norm = 0;
	#pragma omp parallel for
	for (size_t i = 0; i < N; ++i) {
		norm += (Ax[i] * Ax[i]);
	}
	return sqrt (norm);
}

void mulScalarCol (double *Ax, const double scalar,
				   const size_t N) {
	#pragma omp parallel for
	for (size_t i = 0; i < N; ++i) {
		Ax[i] *= scalar;
	}
}

int criterionCompletion (double *Ax, double *b, const size_t N,
						 const double normB, const double EPSILON) {
	difCol (Ax, b, N);
	double firsrNorm = countNorm (Ax, N);
	return (firsrNorm / normB) >= EPSILON;
}

void simpleIterationMethod (double *A, double *b, double *x, const size_t N,
							const double normB, const double EPSILON) {
	const double t = 0.00001;
	double *condition = NULL;
	condition = (double *)malloc (N * sizeof (double));
	double *Ax = NULL;
	Ax = (double *)malloc (N * sizeof (double));

	if (condition == NULL || Ax == NULL) {
		free (Ax);
		free (condition);
		printf ("memory allocation error");
		return;
	}

	mulMatrix (A, x, condition, N);
	int criretion = criterionCompletion (condition, b, N, normB, EPSILON);
	while (criretion) {
		mulMatrix (A, x, Ax, N);
		difCol (Ax, b, N);
		mulScalarCol (Ax, t, N);
		difCol (x, Ax, N);
		mulMatrix (A, x, condition, N);
		criretion = criterionCompletion (condition, b, N, normB, EPSILON);
	}

	free (condition);
	free (Ax);
}

int main (int argc, char *argv[]) {
	const size_t N = 4096;
	const double EPSILON = 0.00000001;
	double startTime;
	double endTime;
	double *A = NULL;
	A = (double *)malloc (N * N * sizeof (double));
	double *b = NULL;
	b = (double *)malloc (N * sizeof (double));
	double *x = NULL;
	x = (double *)malloc (N * sizeof (double));

	if (x == NULL || b == NULL || A == NULL) {
		free (A);
		free (b);
		free (x);
		printf ("Memory allocation error");
		return 0;
	}
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			if (i == j) {
				A[i * N + j] = 2;
			} else {
				A[i * N + j] = 1;
			}
		}
		x[i] = 0;
		b[i] = N + 1;
	}

	startTime = omp_get_wtime();
	double normB = 0;
	#pragma omp parallel for
	for (size_t i = 0; i < N; ++i) {
		normB += (b[i] * b[i]);
	}

	normB = sqrt (normB);
	simpleIterationMethod (A, b, x, N, normB, EPSILON);
	endTime = omp_get_wtime();
	printf("time taken - %f sec\n", endTime - startTime);
	printCol(x, N);
	free (A);
	free (b);
	free (x);
	return 0;
}
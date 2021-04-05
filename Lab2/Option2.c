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
		double normB = 0;

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
		int criretion = 1;
		double firstNorm = 0;
		const double t = 0.00001;
		double *condition = NULL;
		condition = (double *)malloc (N * sizeof (double));
		double *Ax = NULL;
		Ax = (double *)malloc (N * sizeof (double));

		if (condition == NULL || Ax == NULL) {
			free (Ax);
			free (condition);
			printf ("memory allocation error");
			return 0;
		}
		startTime = omp_get_wtime();
		omp_set_num_threads(4);
		#pragma omp parallel shared (A, b, x, criretion, firstNorm,condition,Ax, normB)
	{
		#pragma omp for reduction(+:normB)
			for (size_t i = 0; i < N; ++i)
				normB += (b[i] * b[i]);
		#pragma omp single
		{
			normB = sqrt (normB);
		}
		while (criretion) {

		#pragma omp for reduction(+:firstNorm)
			for (size_t i = 0; i < N; ++i) {
				double cur = 0;
				for (size_t j = 0; j < N; ++j) {
					cur += (A[i * N + j] * x[j]);
				}
				cur -= b[i];
				condition[i] = x[i] - cur * t;
				firstNorm += cur * cur;
			}
		#pragma omp single
			{
				double *tmp = x;
				x = condition;
				condition = tmp;
				firstNorm = sqrt (firstNorm);
				criretion = (firstNorm / normB) > EPSILON;
				firstNorm = 0;
			}
		}

	}
		endTime = omp_get_wtime();
		printf("time taken - %f sec\n", endTime - startTime);
		printCol (x, N);
		free (condition);
		free (Ax);
		free (A);
		free (b);
		free (x);
		return 0;
}
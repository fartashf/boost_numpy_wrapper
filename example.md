# Incomplete simple example.

## Starting with Examples
As our example, we are going to write a code to multiply two matrices. Here is a code that does so in python on two random matrices:

```python
import numpy as np
A = np.random.rand(20,10)
B = np.random.rand(10,30)
C = np.dot(A, B)  # We are going to replace this
```

In the code above, we use the function `dot` that gets two NumPy ndarrays as its arguments and returns an ndarray.

Let's say we already have a C++ function that does this operation. Such a code might use a user-defined matrix class, or a library that provides such a class, or operates directly on flattened arrays. We don't want to change that because it usually takes too much time and we might introduce bugs. Depending on the matrix class used, we might have to copy the arguments' and return values data. It has a little bit of overhead but preferrable to long debugging times.

Here is a code in C++ that computes the result of a matrix multiplication:

```C++
Matrix matrix_mult(Matrix A, Matrix B){
	Matrix C(A.dim(0), B.dim(1));
	int m = A.dim(0), n = B.dim(1), k = A.dim(1);
	for(int i=0; i<m; i++)
		for(int t=0; t<k; t++)
			for(int j=0; j<n; j++)
				c(i, j) += a(i, t)*b(t, j);
	return C;
}
```

In the C++ code above, we assumed that a class called `Matrix` is defined similar to this:

```C++
class Matrix{
	// ...

	public:
		Matrix(int M, int N){
			// Initialize a matrix of size M-by-N
		}
		int dim(int d){
			// Return the size of dimension d
		}
		float *operator()(int i, int j){
			// Return pointer to element at row i, column j
		}
};
```

Later, I will also talk about the case when the C++ code is directly working with flattened representations. In those cases, you have to directly deal with row-major and column-major differences.


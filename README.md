# Boost NumPy Wrapper

Let's say you are writing a code in Python that uses NumPy and you want to use a function written in C++ in your code. The main challenge is that C++ doesn't know about Python/NumPy objects by itself.

Examples are, you want to

- Optimize a function and you know a way to do it in C++ and you don't want to use Cython, PyPI or JIT
- Convert a code written for MATLAB that has some functions written in MEX

The examples in this repo use [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/) and [Boost.NumPy](https://github.com/ndarray/Boost.NumPy) for this purpose. The wrapper facilitates the conversion of the arguments and return values without the need for changing the underlying code.

## Requirements
The following two libraries must be installed:
- Boost compiled with Python support
- Boost.NumPy [Compile_Boost_NumPy](http://www.edge.no/wiki/Compile_Boost_NumPy)

```
git clone https://github.com/ndarray/Boost.NumPy.git
cd Boost.NumPy/
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local/" ..
make; make install
cd ..
mkdir debug
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_DEBUG_POSTFIX="-gd" -DCMAKE_INSTALL_PREFIX="$HOME/.local/" ..
make; make install
```

## Writing a Boost.NumPy Module
- write the wrapper
- bind the wrapper
- write cmakefile
- compile
- import
- compare

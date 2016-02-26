// http://www.edge.no/wiki/Boost_Python
// http://docs.ros.org/hydro/api/boost_numpy/html/classboost_1_1numpy_1_1ndarray.html
#include "wrappers.hpp"
#include <math.h>
#include "string.h"
#include "sse.hpp"
#include <iostream>
#include "gradientMex.cpp"

using namespace std;

#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

// [Gx,Gy] = grad2(I) - see gradient2.m
bp::tuple pGrad2(const np::ndarray& pI)
{
	Py_intptr_t const * strides = pI.get_strides();
	cout << strides[0] << ' ' << strides[1] << endl;
	ASSERT_THROW( (pI.get_nd() == 2 || pI.get_nd() == 3),
			"Expected two or three-dimensional array");
	ASSERT_THROW( (pI.get_dtype() == np::dtype::get_builtin<float>()),
			"Expected array of type double (np.float32)");
	ASSERT_THROW( (pI.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");

	int h=pI.shape(0), w=pI.shape(1), d=(pI.get_nd()==2) ? 1 : pI.shape(2);

	ASSERT_THROW((h>=2 && w>=2),
			"I must be at least 2x2.");

	float *I, *Gx, *Gy;
	I = reinterpret_cast<float*>(pI.get_data());

	np::ndarray pGx = np::from_object(
			np::zeros(bp::make_tuple(h, w, d), np::dtype::get_builtin<float>()),
			np::ndarray::FARRAY);
	np::ndarray pGy = np::from_object(
			np::zeros(bp::make_tuple(h, w, d), np::dtype::get_builtin<float>()),
			np::ndarray::FARRAY);

	Gx = reinterpret_cast<float*>(pGx.get_data());
	Gy = reinterpret_cast<float*>(pGy.get_data());

	grad2(I, Gx, Gy, h, w, d);

	return bp::make_tuple(pGx, pGy);
}

// [M,O] = gradMag( I, channel, full ) - see gradientMag.m
bp::tuple pGradMag(const np::ndarray& pI, int c, int full)
{
	ASSERT_THROW( (pI.get_nd() == 2 || pI.get_nd() == 3),
			"Expected two- or three-dimensional array");
	ASSERT_THROW( (pI.get_dtype() == np::dtype::get_builtin<float>()),
			"Expected array of type double (np.float32)");
	ASSERT_THROW( (pI.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");

	int h=pI.shape(0), w=pI.shape(1), d=(pI.get_nd()==2) ? 1 : pI.shape(2);

	ASSERT_THROW((h>=2 && w>=2),
			"I must be at least 2x2.");
	
	float *I, *M, *O=0;
	I = reinterpret_cast<float*>(pI.get_data());

	if( c>0 && c<=d ) { I += h*w*(c-1); d=1; }

	// TODO: return only one
	np::ndarray pM = np::from_object(
			np::zeros(bp::make_tuple(h, w), np::dtype::get_builtin<float>()),
			np::ndarray::FARRAY);
	np::ndarray pO = np::from_object(
			np::zeros(bp::make_tuple(h, w), np::dtype::get_builtin<float>()),
			np::ndarray::FARRAY);

	M = reinterpret_cast<float*>(pM.get_data());
	O = reinterpret_cast<float*>(pO.get_data());

	gradMag(I, M, O, h, w, d, full>0);

	return bp::make_tuple(pM, pO);
}

// gradMagNorm( M, S, norm ) - operates on M - see gradientMag.m
void pGradMagNorm(const np::ndarray& pM, const np::ndarray& pS, float norm)
{
	ASSERT_THROW( (pM.get_nd() == 2 || pM.get_nd() == 3),
			"Expected two-dimensional array");
	ASSERT_THROW( (pM.get_dtype() == np::dtype::get_builtin<float>()),
			"Expected array of type double (np.float32)");
	ASSERT_THROW( (pM.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");
	ASSERT_THROW( (pS.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");

	int h=pM.shape(0), w=pM.shape(1), d=(pM.get_nd()==2) ? 1 : pM.shape(2);

	ASSERT_THROW((h==pS.shape(0) && w==pS.shape(1) && d==1),
			"M or S is bad.");

	float *M, *S;
	M = reinterpret_cast<float*>(pM.get_data());
	S = reinterpret_cast<float*>(pS.get_data());

	gradMagNorm(M,S,h,w,norm);
}

// H=gradHist(M,O,[...]) - see gradientHist.m
np::ndarray pGradHist(const np::ndarray& pM, const np::ndarray& pO, int binSize=8,
		int nOrients=9, int softBin=1, int useHog=0, float clipHog=0.2f,
		bool full=false)
{
	ASSERT_THROW( (pM.get_nd() == 2 || pM.get_nd() == 3),
			"Expected two- or three-dimensional array");
	ASSERT_THROW( (pM.get_dtype() == np::dtype::get_builtin<float>()),
			"Expected array of type double (np.float32)");
	ASSERT_THROW( (pM.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");
	ASSERT_THROW( (pO.get_flags() & np::ndarray::F_CONTIGUOUS),
			"Expected Fortran-contigiuous array");

	int h=pM.shape(0), w=pM.shape(1), d=(pM.get_nd()==2) ? 1 : pM.shape(2);

	ASSERT_THROW((h==pO.shape(0) && w==pO.shape(1) && d==1),
			"M or O is bad.");
	
	float *M, *O, *H;
	M = reinterpret_cast<float*>(pM.get_data());
	O = reinterpret_cast<float*>(pO.get_data());

	int hb, wb, nChns;
	hb = h/binSize; wb = w/binSize;
	nChns = useHog== 0 ? nOrients : (useHog==1 ? nOrients*4 : nOrients*3+5);
	
	np::ndarray pH = np::from_object(
			np::zeros(bp::make_tuple(hb, wb, nChns), np::dtype::get_builtin<float>()),
			np::ndarray::FARRAY);

	H = reinterpret_cast<float*>(pH.get_data());

	if( nOrients==0 ) return pH;
	if( useHog==0 ) {
		gradHist( M, O, H, h, w, binSize, nOrients, softBin, full );
	} else if(useHog==1) {
		hog( M, O, H, h, w, binSize, nOrients, softBin, full, clipHog );
	} else {
		fhog( M, O, H, h, w, binSize, nOrients, softBin, clipHog );
	}
	return pH;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(pGradHist_overloads, pGradHist, 2, 8)

BOOST_PYTHON_MODULE(gradientMex)
{
	np::initialize();

	bp::def("gradient2", pGrad2);
	bp::def("gradientMag", pGradMag);
	bp::def("gradientMagNorm", pGradMagNorm);
	bp::def("gradientHist", pGradHist, pGradHist_overloads());
}

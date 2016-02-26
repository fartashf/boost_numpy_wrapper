// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example shows how to use the correlation_tracker from the dlib C++ library.  This
    object lets you track the position of an object as it moves from frame to frame in a
    video sequence.  To use it, you give the correlation_tracker the bounding box of the
    object you want to track in the current video frame.  Then it will identify the
    location of the object in subsequent frames.

    In this particular example, we are going to run on the video sequence that comes with
    dlib, which can be found in the examples/video_frames folder.  This video shows a juice
    box sitting on a table and someone is waving the camera around.  The task is to track the
    position of the juice box as the camera moves around.
*/

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include "conversion.h"

using namespace dlib;
using namespace std;

/*VVVVV Boost python and numpy stuff VVVVV*/

#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

/*^^^^^ Boost python and numpy stuff ^^^^^*/


class correlation_tracker_py{
	private:
		correlation_tracker tracker;

	public:
		// Create a tracker and start a track on
		// the given bounding box
		void start_track(boost::python::object &pyimg,
				const rectangle& r){
			//if(is_gray_python_image(img)){
			//}
			array2d<rgb_pixel> img;
			pyimage_to_dlib_image(pyimg, img);
			tracker.start_track(img, r);
		}
		// Keep calling update with new frames
		double update(const boost::python::object &pyimg){
			array2d<rgb_pixel> img;
			pyimage_to_dlib_image(pyimg, img);
			return tracker.update(img);
		}
		// Get the position in the last frame
		rectangle get_position(){
			return tracker.get_position();
		}
};

void bind_correlation_tracker(){
	class_<correlation_tracker_py>("correlation_tracker")
		.def("start_track", &correlation_tracker_py::start_track)
		.def("update", &correlation_tracker_py::update)
		.def("get_position", &correlation_tracker_py::get_position)
		;
}

import dlib
from skimage import io
import time


tic = time.time()

# Start the tracker, give it the initial bounding box
img = io.imread('../examples/video_frames/frame_000100.jpg')
tracker = dlib.correlation_tracker()
center = [93, 110]
w_h = [38, 86]
r = dlib.rectangle(center[0]-w_h[0]/2,
                   center[1]-w_h[1]/2,
                   center[0]+w_h[0]/2,
                   center[1]+w_h[1]/2)
tracker.start_track(img, r)

# Iterate over frames
for i in range(101, 250):
    img = io.imread('../examples/video_frames/frame_000%d.jpg' % i)
    tracker.update(img)
    print tracker.get_position()

toc = time.time()

print 'Elapsed time: %.2fs' % (toc-tic)

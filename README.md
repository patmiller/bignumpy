# bignumpy
File backed numpy array objects for python

I built this for a friend who needed to scale a a computation out on his memory challenged laptop with no access to a cloud computer.  Essentially, this converted Numpy arrays into mmap/shared memory backed values.  Here, it was OK to run at "disk" speeds as memory swapped in as needed to make the computation.  

#!/usr/bin/python

'''

HaloStack v0.1 by Panu Lahtinen (April 2012)

For more information, see http://www.iki.fi/~pnuu/halostack/

'''

from PythonMagick import Image as PMImage
from PythonMagick import Blob 
import numpy as np
from optparse import OptionParser
import os.path
import sys
import time
from glob import glob

def IMtoNumpy(img, usm=None, verbose=False):
    if verbose:
        print "Converting image to numpy array"

    # Check if USM sharpening should be used
    if usm != None:
        if verbose:
            print "Running unsharp mask filter"
        r,s,a,t = (usm)
        img.unsharpmask(r,s,a,t)

    # Convert to RGB and get the pixel data as characters
    if verbose:
        print "Converting to RGB"
    img.magick('RGB')
    blb = Blob()
    img.write(blb)
    data = blb.data
    # Check what bit-depth the image has and convert to Numpy array
    if img.depth() == 8:
        n_im = np.fromstring(data, dtype='uint8') #.astype('uint16')
    else:
        n_im = np.fromstring(data, dtype='uint16')

    # Reshape the image from vector to height x width x channels
    h = img.rows()
    w = img.columns()
    c = 3
    if img.monochrome():
        c = 1

    n_im = n_im.reshape(h,w,c)
    return n_im

def NumpytoIM(img, usm=None, verbose=False):
    if verbose:
        print "Converting numpy array to ImageMagick"

    out_img = PMImage()
    if img.dtype == 'uint16':
        out_img.depth(16)
    else:
        out_img.depth(8)
    out_img.magick('RGB')
    h,w,c = img.shape
    size_str = str(w)+'x'+str(h)
    out_img.size(size_str)

    b = Blob()
    b.data = img.tostring()
    out_img.read(b)
    out_img.magick('PNG')

    # Check if USM sharpening should be used
    if usm != None:
        if verbose:
            print "Running unsharp mask filter"
        r,s,a,t = (usm)
        out_img.unsharpmask(r,s,a,t)

    return out_img

def parse_usm(usm, u):
    usm = usm.split(',')
    if len(usm) < 2 or len(usm) > 4:
        print "Unsharp mask parameters should be given as:"
        print "-%s radius,amount (-%s 50,4) OR" % (u,u)
        print "-%s radius,amount,sigma (-%s 30,1.1,10) OR" % (u,u)
        print "-%s radius,amount,sigma,threshold (-%s 25,2.1,5,10)" % (u,u)
        print "If omitted, sigma defaults to radius/2 and threshold defaults to 0"
        sys.exit()
    # Radius
    r = int(usm[0])
    # Amount
    a = float(usm[1])
    # Sigma
    if(len(usm) > 2):
        s = int(usm[2])
    else:
        s = r/2
    # Threshold
    if(len(usm) > 3):
        t = int(usm[3])
    else:
        t = 0

    return (r,s,a,t)

# Main program starts here

start_time = time.time()

usage = "usage: %prog [options] img1 img2 ... imgn"
parser = OptionParser(usage=usage)
parser.add_option("-a", "--average-stack", dest="avg_stack_file", 
                  default=None, metavar="FILE",
                  help="Filename of the average stack [default: not saved]")
parser.add_option("-m", "--max-intensity-stack", dest="max_int_stack_file", 
                  default=None, metavar="FILE",
                  help="Filename of the maximum intensity stack [default: not saved]")
parser.add_option("-M", "--max-deviation-stack", dest="max_dev_stack_file", 
                  default=None, metavar="FILE",
                  help="Filename of the maximum RGB-deviation stack [default: not saved]")
parser.add_option("-c", "--max-color-diff", dest="max_col_diff_stack_file",
                  default=None, metavar="FILE",
                  help="Filename of the maximum color separation stack [default: not saved]")
parser.add_option("-e", "--median", dest="median_stack_file", default=None,
                  metavar="FILE", help="Filename of the median stack [default: not saved]")
parser.add_option("-u", "--pre-usm", dest="pre_usm", default=None,
                  metavar="R,A[,S[,T]]", help="Unsharp mask parameters applied before stacking [default: None]")
parser.add_option("-U", "--post-usm", dest="post_usm", default=None,
                  metavar="R,A[,S[,T]]", help="Unsharp mask parameters applied after stacking [default: None]")
parser.add_option("-b", "--base-image", dest="base_image", default=None,
                  metavar="FILE", help="Use this image file as base for alignment and image intensity normalization")
parser.add_option("-n", "--normalize", dest="normalization", default=None,
                  metavar="NUM", help="Normalize images to given value, or to first image (-n 0) before stacking [default: no normalization]")
parser.add_option("-N", "--normalize-area", dest="normalization_area",
                  default=None, metavar="X1,Y1,X2,Y2", help="Area used to determine the normalization [default: full image]")
parser.add_option("-r", "--align-reference", dest="align_ref_loc",
                  default=None, metavar="x,y,s", help="Center and size of square used as reference area for image alignment [default: no image alignment]")
parser.add_option("-s", "--align-search-area", dest="align_search_area",
                  default=None, metavar="X1,Y1,X2,Y2", help="Area to be searched for image alignment point")
parser.add_option("-t", "--correlation-threshold", dest="correlation_threshold", 
                  default=0.0, metavar="NUM", help="Minimum alignment correlation required to include the image for stacking [default: %default]")
parser.add_option("-R", "--rotate-points", dest="rotate_points",
                  default=None, metavar="X1,Y2,X2,Y2,S", help="Define two square-shaped areas (2S+1 x 2S+1) that are used to rotate images so that these points are on a horizontal line")
parser.add_option("-O", "--output-images", dest="output_images",
                  default=None, metavar="STR", help="Save images as PNG with given filename postfix [default: not saved]")
parser.add_option("-v", "--verbose", dest="verbose", default=False,
                  action="store_true", help="Enable verbose output")

# Parse commandline inputs
(options, args) = parser.parse_args()

avg_stack_file = options.avg_stack_file
max_int_stack_file = options.max_int_stack_file
max_dev_stack_file = options.max_dev_stack_file
max_col_diff_stack_file = options.max_col_diff_stack_file
median_stack_file = options.median_stack_file
normalization = options.normalization
normalization_area = options.normalization_area

if normalization is not None:
    normalization = int(normalization)
    if normalization_area is not None:
        normalization_area = normalization_area.split(',')
        if len(normalization_area) != 4:
            print "Wrong number of parameters for normalization area."
            print "Use -E x1,y1,x2,y2 eg. -E 494,234,504,247"
            sys.exit()
        x1,y1,x2,y2 = normalization_area
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        normalization_area = x1,y1,x2,y2
            
align_ref_loc = options.align_ref_loc
if align_ref_loc is not None:
    align_ref_loc = align_ref_loc.split(',')
    if len(align_ref_loc) != 3:
        print "Wrong number of parameters for alignment reference location and size."
        print "Use -r x,y,d  eg. -r 430,234,7  which defines a square from 423,227 to 437,241"
        sys.exit()
    align_ref_loc = [int(align_ref_loc[0]),int(align_ref_loc[1]),int(align_ref_loc[2])]
    # Import math library needed for alignment calculations
    import math
    # Import alignment code
    from align_functions import *

align_search_area = options.align_search_area
if align_search_area is not None:
    align_search_area = align_search_area.split(',')
    if len(align_search_area) != 4:
        print "Wrong number of parameters for alignment search area."
        print "Use -s x1,y1,x2,y2 eg. -s 494,234,504,247"
        sys.exit()
    sx1,sy1,sx2,sy2 = align_search_area
    align_search_area = (int(sx1),int(sy1),int(sx2),int(sy2))
correlation_threshold = float(options.correlation_threshold)
output_images = options.output_images
pre_usm = options.pre_usm
if pre_usm is not None:
    pre_usm = parse_usm(pre_usm, 'u')
post_usm = options.post_usm
if post_usm is not None:
    post_usm = parse_usm(post_usm, 'U')
base_image = options.base_image
rotate_points = options.rotate_points
if rotate_points is not None:
    from scipy import ndimage
    rotate_points = rotate_points.split(',')
    if len(rotate_points) != 5:
        print "Wrong number of parameters for rotation estimation."
        print "Use -R rx1,ry1,rx2,ry2,rs eg. 860,1800,2500,1840,25"
        sys.exit()
    rx1,ry1,rx2,ry2,rs = rotate_points
    rotate_points = (int(rx1),int(ry1),int(rx2),int(ry2),int(rs))

verbose = options.verbose    
frames_in = args
frames = []

# Ensure that all files are used also on Windows, as the command
# prompt does not automatically parse wildcards to a list of images
for f in frames_in:
    if '*' in f:
        f_all = glob(f)
        for f2 in f_all:
            frames.append(f2)
            if verbose:
                print "Added %s to the image list" % f2
    else:
        frames.append(f)
        if verbose:
            print "Added %s to the image list" % f

# If nothing is to be saved, or no input images are supplied, abort
if avg_stack_file is None and max_int_stack_file is None and max_dev_stack_file is None and max_col_diff_stack_file is None and median_stack_file is None:
    print "No stacks will be saved, aborting."
    print ""
    parser.print_help()
    sys.exit()
n_files = len(frames)
if n_files == 0:
    print "No input images, aborting."
    print ""
    parser.print_help()
    sys.exit()

print "%d files found" % n_files

# Set the named file as first entry in the image list
# This file will be the base for alignment and intensity normalization
if base_image is not None:
    idx = frames.index(base_image)
    f = frames.pop(idx)
    frames.insert(0,f)

if verbose:
    print "%s will be used as base image" % frames[0]

# Determine image size
base_img_fname = frames[0]
if verbose:
    print "Reading %s" % base_img_fname
img = PMImage(base_img_fname)
w,h = img.size().width(), img.size().height()
if img.monochrome():
    c = 1
else:
    c = 3

print "Detected image size is %d x %d (width x height)" % (w,h)

print "%s will be used as base image" % frames[0]
img = IMtoNumpy(img,pre_usm,verbose=verbose)

# Get data used for aligning images
align_ref_data = None
if align_ref_loc is not None:
    xc = align_ref_loc[0]
    yc = align_ref_loc[1]
    d = align_ref_loc[2]
    align_ref_data = img[yc-d:yc+d+1, xc-d:xc+d+1,:].mean(axis=2)

# Get data used for co-rotate the images
if rotate_points is not None:
    rx1,ry1,rx2,ry2,rs = rotate_points
    rot1_data = img[ry1-rs:ry1+rs+1,rx1-rs:rx1+rs+1,:].mean(axis=2)
    rot2_data = img[ry2-rs:ry2+rs+1,rx2-rs:rx2+rs+1,:].mean(axis=2)
    rotate_angle = 180*math.atan2((ry2-ry1),(rx2-rx1))/math.pi
    if rotate_angle is not 0:
        print "Rotation angle is %3.3f deg" % rotate_angle
        img = ndimage.rotate(img, rotate_angle, reshape=False)
        # If the base-image got rotated, we need to align it so
        # that the reference point is in the correct place
        b,xb,yb = align(img.mean(axis=2), align_ref_data, area=align_search_area)
        tx = align_ref_loc[0] - xb # translation in x-direction
        ty = align_ref_loc[1] - yb # translation in y-direction
        print "Shift: x: %d, y: %d, correlation (R^2): %1.3f" % (tx,ty,b)
        n_x1,n_x2,n_y1,n_y2,o_x1,o_x2,o_y1,o_y2 = calc_align_area(tx,ty,w,h) 
        # Move image data by the calculated amount
        n_img = 0*img.copy()
        n_img[n_y1:n_y2,n_x1:n_x2,:] = img[o_y1:o_y2,o_x1:o_x2,:]
        img = n_img.copy()

# Calculate normalization value, or normalize the base image to given value
if normalization is not None:
    if normalization_area is None:
        normalization_area = (0,0,w,h)
    x1,y1,x2,y2 = normalization_area

    if normalization > 0:
        if verbose:
            print "Normalizing to %d" % normalization
        img_tmp = img*(1.0*normalization/np.median(img[y1:y2,x1:x2,:].mean(axis=2)))
        if img.dtype == 'uint16':
            img_tmp[img_tmp > 2**16 - 1] = 2**16 - 1
        else:
            img_tmp[img_tmp > 2**8 - 1] = 2**8 - 1
        img = img_tmp.astype(img.dtype)
        del img_tmp
    if normalization == 0:
        if verbose:
            print "Calculating normalization value"
        normalization = np.median(img[y1:y2,x1:x2,:].mean(axis=2))

# Save image as PNG
if output_images is not None:
    out_fname = '.'.join(base_img_fname.split('.')[0:-1]) # get rid of extension
    out_fname = "%s_%s.png" % (out_fname,output_images)
    print "Saving image: %s" % out_fname
    out_img = NumpytoIM(img, None)
    out_img.write(out_fname)

# Create output arrays for selected image stacks
if verbose:
    print "Initialize stacks"
avg_stack = None
max_int_stack = None
max_dev_stack = None
max_col_diff_stack = None
median_stack = None
if avg_stack_file is not None:
    avg_stack = np.zeros((h,w,c), dtype='double')
    avg_stack = avg_stack + img
if max_int_stack_file is not None:
    max_int_stack = np.zeros((h,w,c), dtype=img.dtype)
    max_int_stack = max_int_stack + img
    # Maximum intensities need to be 32-bit to prevent overflow
    max_pix_intensities = np.zeros((h,w), dtype='uint32')
    max_pix_intensities = max_pix_intensities + img.sum(axis=2)
if max_dev_stack_file is not None:
    max_dev_stack = np.zeros((h,w,c), dtype=img.dtype)
    max_dev_stack = max_dev_stack + img
    max_pix_deviations = np.zeros((h,w), dtype='uint16')
    max_pix_deviations = max_pix_deviations + img.std(axis=2)
if max_col_diff_stack_file is not None:
    max_col_diff_stack = np.zeros((h,w,c), dtype=img.dtype)
    max_col_diff_stack = max_col_diff_stack + img
    max_col_diffs = img.max(axis=2)-img.min(axis=2)
if median_stack_file is not None:
    # Import HDF5 library
    import h5py
    median_stack = np.zeros((h,w,c), dtype=img.dtype)
    # Create a HDF5 file for temporary storage space for median stack
    # calculations. The file is memorymapped, so access from the
    # program is easy, but amount of disk-I/O will be huge.
    median_hdf_file = median_stack_file[:-3]
    median_hdf_file = median_hdf_file + 'H5'
    median_hdf_fid = h5py.File(median_hdf_file,'w')
    # Datasets will be h-high, w-wide and n_files-deep with the same
    # bit-depth as the input images
    hdf_R = median_hdf_fid.create_dataset('R', (h,w,n_files), dtype=img.dtype)
    hdf_G = median_hdf_fid.create_dataset('G', (h,w,n_files), dtype=img.dtype)
    hdf_B = median_hdf_fid.create_dataset('B', (h,w,n_files), dtype=img.dtype)
    # Put the data from the first image to the HDF5 file
    hdf_R[:,:,0] = img[:,:,0]
    hdf_G[:,:,0] = img[:,:,1]
    hdf_B[:,:,0] = img[:,:,2]

# Remove the first frame from the list, as it has been already used
frames = frames[1:]

frames_used=1 # number of images used in the stacks
for f in frames:
    if os.path.isfile(f):
        print f
        # Read image to numpy array
        img = IMtoNumpy(PMImage(f),pre_usm)

        # Calculate rotation and rotate the image
        # Calculations for rotation angle not yet implemented
        if rotate_points is not None:
            rx1,ry1,rx2,ry2,rs = rotate_points
            b1,x1b,y1b = align(img.mean(axis=2), rot1_data, area=(rx1-100,ry1-100,rx1+100,ry1+100))
            b2,x2b,y2b = align(img.mean(axis=2), rot2_data, area=(rx2-100,ry2-100,rx2+100,ry2+100))
            rotate_angle = 180*math.atan2((y2b-y1b),(x2b-x1b))/math.pi
            if rotate_angle is not 0:
                print "Rotation angle: %3.3f deg with correlations of %1.3f and %1.3f" % (rotate_angle,b1,b2)
                img = ndimage.rotate(img, rotate_angle, reshape=False)

        # Find alignment point and get translation parameters, adjust
        if align_ref_data is not None:
            b,xb,yb = align(img.mean(axis=2), align_ref_data, area=align_search_area)
            tx = align_ref_loc[0] - xb # translation in x-direction
            ty = align_ref_loc[1] - yb # translation in y-direction
            print "Shift: x: %d, y: %d, correlation (R^2): %1.3f" % (tx,ty,b)

            # If correlation between alignment points is too low, skip
            # this frame
            if b < correlation_threshold:
                print "Correlation below threshold (%1.3f < %1.3f), skipping file %s" % (b,correlation_threshold,f)
                continue

            n_x1,n_x2,n_y1,n_y2,o_x1,o_x2,o_y1,o_y2 = calc_align_area(tx,ty,w,h) 

            # Move image data by the calculated amount
            n_img = 0*img.copy()
            n_img[n_y1:n_y2,n_x1:n_x2,:] = img[o_y1:o_y2,o_x1:o_x2,:]
            img = n_img.copy()
            del n_img

        # Normalize image intensity
        if normalization is not None:
            # Normalize the value within given image region. This has
            # the benefit that only the important area has any say on
            # the normalization, and the unimportant areas are ignored
            if verbose:
                print "Normalizing image intensity to %d" % normalization
            x1,y1,x2,y2 = normalization_area
            img_tmp = img*(1.0*normalization/np.median(img[y1:y2,x1:x2,:].mean(axis=2)))
            if img.dtype == 'uint16':
                img_tmp[img_tmp > 2**16 - 1] = 2**16 - 1
            else:
                img_tmp[img_tmp > 2**8 - 1] = 2**8 - 1
            img = img_tmp.astype(img.dtype)
            del img_tmp

        # Save image as png
        if output_images is not None:
            out_fname = '.'.join(f.split('.')[0:-1]) # get rid of extension
            out_fname = "%s_%s.png" % (out_fname,output_images)
            print "Saving image: %s" % out_fname
            out_img = NumpytoIM(img, None)
            out_img.write(out_fname)
            del out_img

        # Add data to cumulative sum array, which will be used to
        # calculate average image
        if avg_stack is not None:
            if verbose:
                print "Updating average stack"
            avg_stack = avg_stack + img
        
        # Update pixels with greatest intensities
        if max_int_stack is not None:
            if verbose:
                print "Updating maximum intensity stack"
            intensities = img.sum(axis=2)
            idxs = intensities > max_pix_intensities
            max_pix_intensities[idxs] = intensities[idxs]
            max_int_stack[idxs,0] = img[idxs,0]
            max_int_stack[idxs,1] = img[idxs,1]
            max_int_stack[idxs,2] = img[idxs,2]
            del intensities
            del idxs
        # Update pixels with greatest standard deviation in color values
        if max_dev_stack is not None:
            if verbose:
                print "Updating maximum color deviation stack"
            deviations = img.std(axis=2)
            idxs = deviations > max_pix_deviations
            max_pix_deviations[idxs] = deviations[idxs]
            max_dev_stack[idxs,0] = img[idxs,0]
            max_dev_stack[idxs,1] = img[idxs,1]
            max_dev_stack[idxs,2] = img[idxs,2]
            del deviations
            del idxs
        # Update pixels with greatest separation between brightest and
        # dimmest color channel
        if max_col_diff_stack is not None:
            if verbose:
                print "Updating maximum color deviation stack"
            diff = img.max(axis=2)-img.min(axis=2)
            idxs = diff > max_col_diffs
            max_col_diffs[idxs] = diff[idxs]
            max_col_diff_stack[idxs,0] = img[idxs,0]
            max_col_diff_stack[idxs,1] = img[idxs,1]
            max_col_diff_stack[idxs,2] = img[idxs,2]
            del diff
            del idxs
        # Add image data to a HDF5 file used as data-cube. This file
        # is used as a temporary memory-mapped storage when
        # calculating median of all the co-aligned image pixels
        if median_stack is not None:
            if verbose:
                print "Updating median stack"
            hdf_R[:,:,frames_used] = img[:,:,0]
            hdf_G[:,:,frames_used] = img[:,:,1]
            hdf_B[:,:,frames_used] = img[:,:,2]

        # Increment the frame counter. This happens only if the
        # correlation has been good enough in alignment
        frames_used += 1
            
    else:
        print "File %s does not exist, skipping" % f

del img

# Scale and save average stack
if avg_stack is not None:
    if verbose:
        print "Finishing average stack:"
        print "- subtract minimum value"
    # Image is scaled between 0 and 2^16 -1 to cover the whole 16-bit/color
    # range of the output image file
    avg_stack = avg_stack - avg_stack.min()
    if verbose:
        print "- scale to 0 ... 65535"
    avg_stack = ((2**16-1)*avg_stack/avg_stack.max()).astype('uint16')
    avg_stack = NumpytoIM(avg_stack, post_usm)
    if verbose:
        print "- saving average stack"
    avg_stack.write(avg_stack_file)
    print "%s saved" % avg_stack_file

# Save maximum intensity stack
if max_int_stack is not None:
    max_int_stack = NumpytoIM(max_int_stack, post_usm)
    max_int_stack.write(max_int_stack_file)
    print "%s saved" % max_int_stack_file

# Save maximum color deviation stack
if max_dev_stack is not None:
    max_dev_stack = NumpytoIM(max_dev_stack, post_usm)
    max_dev_stack.write(max_dev_stack_file)
    print "%s saved" % max_dev_stack_file

# Save maximum color separation stack
if max_col_diff_stack is not None:
    max_col_diff_stack = NumpytoIM(max_col_diff_stack, post_usm)
    max_col_diff_stack.write(max_col_diff_stack_file)
    print "%s saved" % max_col_diff_stack_file

# Save median stack
if median_stack is not None:
    print "Calculating median:"
    # Calculate median for valid data.
    if verbose:
        print "- red channel"
    median_stack[:,:,0] = np.median(hdf_R[:,:,0:frames_used], axis=2)
    if verbose:
        print "- green channel"
    median_stack[:,:,1] = np.median(hdf_G[:,:,0:frames_used], axis=2)
    if verbose:
        print "- blue channel"
    median_stack[:,:,2] = np.median(hdf_B[:,:,0:frames_used], axis=2)
    if verbose:
        print "- closing HDF5 file %s" % median_hdf_file
    median_hdf_fid.close()
    print "Median stack calculated"
    median_stack = NumpytoIM(median_stack, post_usm)
    if verbose:
        print "Writing image"
    median_stack.write(median_stack_file)
    print "%s saved" % median_stack_file

t = time.time() - start_time
hours = int(t/3600)
minutes = int((t-hours*3600)/60)
seconds = int(t-hours*3600-minutes*60)
print "All stacks created and saved from %d/%d images in %d h %d min %3.2f sec" % (frames_used,n_files,hours,minutes,seconds)
print ""

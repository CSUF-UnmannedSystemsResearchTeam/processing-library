#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Image Processing Utilities

Copyright 2017 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import pyexiv2
from PIL import Image
import glob
import metadata
import os
import tifffile as tiff

def raw_image_to_radiance(meta, imageRaw):
   
    # get image dimensions
    imageRaw = imageRaw.T
    xDim = imageRaw.shape[0]
    yDim = imageRaw.shape[1]

    #  get radiometric calibration factors

    # radiometric sensitivity
    a1, a2, a3 = meta.get_item('Xmp.MicaSense.RadiometricCalibration')
    a1 = float(a1)
    a2 = float(a2)
    a3 = float(a3)

    # get dark current pixel values
    # get number of stored values
    black_levels = [float(val) for val in meta.get_item('Exif.Image.BlackLevel')]
    blackLevel = np.array(black_levels)
    darkLevel = blackLevel.mean()

    #get exposure time & gain (gain = ISO/100)
    exposureTime = float(meta.get_item('Exif.Photo.ExposureTime'))
    gain = float(meta.get_item('Exif.Photo.ISOSpeed'))/100.0 


    # apply image correction methods to raw image
    # step 1 - row gradient correction, vignette & radiometric calibration:
    # compute the vignette map image
    V, x, y = vignette_map(meta, xDim, yDim)

    # row gradient correction
    R = 1.0 / (1.0 + a2 * y / exposureTime - a3 * y)

    # subtract the dark level and adjust for vignette and row gradient
    L = V * R * (imageRaw - darkLevel)

    # Floor any negative radiances to zero (can happend due to noise around blackLevel)
    L[L < 0] = 0

    #L = np.round(L).astype(np.uint16)

    # apply the radiometric calibration - i.e. scale by the gain-exposure product and
    # multiply with the radiometric calibration coefficient
    # need to normalize by 2^16 for 16 bit images
    # because coefficients are scaled to work with input values of max 1.0
    bitsPerPixel = meta.get_item('Exif.Image.BitsPerSample')
    bitDepthMax = float(2**bitsPerPixel)
    radianceImage = L.astype(float)/(gain * exposureTime)*a1/bitDepthMax
    
    # return both the radiance compensated image and the DN corrected image, for the
    # sake of the tutorial and visualization
    return radianceImage.T, L.T, V.T, R.T

def vignette_map(meta, xDim, yDim):
	
    # get vignette center
    xVignette = float(meta.get_item('Xmp.Camera.VignettingCenter', 0))
    yVignette = float(meta.get_item('Xmp.Camera.VignettingCenter', 1))

    # get vignette polynomial
    NvignettePoly = meta.size('Xmp.Camera.VignettingPolynomial')
    vignettePolyList = [float(meta.get_item('Xmp.Camera.VignettingPolynomial', i)) for i in range(NvignettePoly)]

    # reverse list and append 1., so that we can call with numpy polyval
    vignettePolyList.reverse()
    vignettePolyList.append(1.)
    vignettePoly = np.array(vignettePolyList)

    # perform vignette correction
    # get coordinate grid across image
    x, y = np.meshgrid(np.arange(xDim), np.arange(yDim))

    #meshgrid returns transposed arrays
    x = x.T
    y = y.T

    # compute matrix of distances from image center
    r = np.hypot((x-xVignette), (y-yVignette))

    # compute the vignette polynomial for each distance - we divide by the polynomial so that the
    # corrected image is image_corrected = image_original * vignetteCorrection
    vignette = 1./np.polyval(vignettePoly, r)
    return vignette, x, y

def radiance_to_reflectance(bandName, radianceImage, imageRaw):
	
	markedImg = radianceImage.copy()
	
	ulx = 624 #Upper left column(x coordinate) of panel area
	uly = 488 #Upper left row(y coordinate) of panel area
	lrx = 864 #Lower right column(x coordinate) of panel area
	lry = 866 #Lower right row (y coordinate) of panel area
	#cv2.rectangle(markedImg, (ulx, uly), (lrx, lry), (0, 255, 0), 3)
	
	# Our panel calibration by band
	panelCalibration = {
		"Blue": 0.4925,
		"Green": 0.4929,
		"Red": 0.4912,
		"NIR": 0.4875,
		"Red Edge": 0.4900
	}
	
	# Select panel region from radiance image
	panelRegion = radianceImage[uly:lry, ulx:lrx]
	
	meanRadiance = panelRegion.mean()
	
	panelReflectance = panelCalibration[bandName]
	radianceToReflectance = panelReflectance / meanRadiance
	reflectanceImage = radianceImage * radianceToReflectance
	anelRegionRaw = imageRaw[uly:lry, ulx:lrx]
	panelRegionRefl = reflectanceImage[uly:lry, ulx:lrx]
	panelRegionReflBlur = cv2.GaussianBlur(panelRegionRefl,(55,55),5)
	#plotutils.plotwithcolorbar(panelRegionReflBlur, 'Smoothed panel region in reflectance image')
	print('Min Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.min()))
	print('Max Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.max()))
	print('Mean Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.mean()))
	print('Standard deviation in region: {:1.4f}'.format(panelRegionRefl.std()))

	return reflectanceImage

def correct_lens_distortion(meta, image):
    # get lens distortion parameters
    Ndistortion = meta.size('Xmp.Camera.PerspectiveDistortion')
    distortionParameters = np.array([float(meta.get_item('Xmp.Camera.PerspectiveDistortion', i)) for i in range(Ndistortion)])
    #get the two principal points
    pp = np.array(meta.get_item('Xmp.Camera.PrincipalPoint').split(',')).astype(np.float)
    # values in pp are in [mm] and need to be rescaled to pixels
    FocalPlaneXResolution = float(meta.get_item('Exif.Photo.FocalPlaneXResolution'))
    FocalPlaneYResolution = float(meta.get_item('Exif.Photo.FocalPlaneYResolution'))

    cX = pp[0] * FocalPlaneXResolution
    cY = pp[1] * FocalPlaneYResolution
    #k = distortionParameters[0:3] # seperate out k -parameters
    #p = distortionParameters[3::] # separate out p - parameters
    #fx = fy = float(meta.get_item('Xmp.Camera.PerspectiveFocalLength'))
    fx = fy = meta.focal_length_mm() * FocalPlaneXResolution
    # apply perspective distortion

    h, w = image.shape

    # set up camera matrix for cv2
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = fx
    cam_mat[1, 1] = fy
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = cX
    cam_mat[1, 2] = cY

    # set up distortion coefficients for cv2
    #dist_coeffs = np.array(k[0],k[1],p[0],p[1],k[2]])
    dist_coeffs = distortionParameters[[0, 1, 3, 4, 2]]
    
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(cam_mat,
                                             dist_coeffs,
                                             np.eye(3),
                                             new_cam_mat,
                                             (w, h),
                                             cv2.CV_32F) # cv2.CV_32F for 32 bit floats
    # compute the undistorted 16 bit image
    undistortedImage = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return undistortedImage

def copyMetadata(imgSource, imgDest):
	
	#Copies over all metadata from one image to another
	
	metaSource = pyexiv2.ImageMetadata(imgSource)
	metaSource.read()
	
	metaDest = pyexiv2.ImageMetadata(imgDest)
	metaDest.read()
	metaSource.copy(metaDest, comment=False)
	metaDest.write()
	
	return

def correctImages(inputPath, outputPath):
	
	# Reads all tiff images in an input folder
	# Correct the distortion
	# Then writes the undistorted image to the output folder
	# Also copies the metadata from the raw image to the undistorted image
	image_list = []
	for filename in glob.glob(inputPath + '/*.tif'):
		image_list.append(filename)
	
	for item in sorted(image_list):
		
		print(item)
		fileItem = os.path.basename(item)
		print(fileItem)
		print(outputPath + fileItem)
		
		meta = metadata.Metadata(item)
		flightImageRaw = cv2.imread(item, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		flightUndistorted = correct_lens_distortion(meta, flightImageRaw)
		resultPath = outputPath + fileItem
		tiff.imsave(resultPath, flightUndistorted)
		copyMetadata(item, resultPath)
		
	return

def correctRaReDe(inputPath, outputPath):
	
	# Reads all the tiff images in an input folder
	# Converts each images to radiance
	# Converts radiance to reflectance
	# Converts reflectance to undistorted image
	
	# Unfinished -- causes the undistorted image to be quite a bit darker
	# Than it otherwise would be
	# 
	
	image_list = []
	for filename in glob.glob(inputPath + '/*.tif'):
		image_list.append(filename)
	
	for item in sorted(image_list):
		
		print(item)
		fileItem = os.path.basename(item)
		print(fileItem)
		print(outputPath + fileItem)
		
		meta = metadata.Metadata(item)
		flightImageRaw = cv2.imread(item, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		flightUndistorted = correct_lens_distortion(meta, flightImageRaw)
		resultPath = outputPath + fileItem
		tiff.imsave(resultPath, flightUndistorted)
		copyMetadata(item, resultPath)
		
	return
	
	flightRadianceImage, _, _, _ = utils.raw_image_to_radiance(meta, flightImageRaw)
	plotutils.plotwithcolorbar(flightRadianceImage, 'Radiance Image')
	radianceToReflectance = utils.radiance_to_reflectance(meta.band_name(), flightRadianceImage, flightImageRaw)
	print(radianceToReflectance)
	flightReflectanceImage = flightRadianceImage * radianceToReflectance
	plotutils.plotwithcolorbar(flightReflectanceImage, 'Reflectance converted')
	flightUndistortedReflectance = utils.correct_lens_distortion(meta, flightReflectanceImage)
	fUR = flightUndistortedReflectance * 255
	
	#flightUndistortedReflectance
	#plt.imsave('Result.tiff', flightUndistortedReflectance)
	#plt.imsave("Result.tiff", fUR)
	plotutils.plotwithcolorbar(flightUndistortedReflectance, 'Reflectance converted and undistorted')
	tiff.imsave('RadReflUndis.tif', flightUndistortedReflectance)

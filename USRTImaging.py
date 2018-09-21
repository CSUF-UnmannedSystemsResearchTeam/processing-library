#!/usr/bin/env python3

#Image, raster, and number processing libraries
import cv2
import numpy as np
import os, sys
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.image as mpimg
import subprocess
from pathlib import Path
import math
import gdal
from gdalconst import *
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
from osgeo import gdal


#File handling and processing libraries
import subprocess
import os.path
import glob
import shutil
from libtiff import TIFF


def getGPSCoordinates(filename):

	'''
	Grab GPS latitude, longitude and their respective reference
	directions from the given image file
	
	Returns the longitude, latitude in degrees as a tuple
	'''

	metadata = pyexiv2.ImageMetadata(filename)
	metadata.read()
	tagLat = metadata["Exif.GPSInfo.GPSLatitude"]
	tagLatRef = metadata["Exif.GPSInfo.GPSLatitudeRef"]
	tagLong = metadata["Exif.GPSInfo.GPSLongitude"]
	tagLongRef = metadata["Exif.GPSInfo.GPSLongitudeRef"]
	
	lad = float(tagLat.value[0])
	lam = float(tagLat.value[1])
	las = float(tagLat.value[2])
	latitude = lad + (lam/60.0) + (las/3600.0)
	
	lod = float(tagLong.value[0])
	lom = float(tagLong.value[1])
	los = float(tagLong.value[2])
	longitude = lod + (lam/60.0) + (los/3600.0)
	
	if(tagLatRef.value == "S"):
		latitude = latitude * -1
	if(tagLongRef.value == "W"):
		longitude = longitude * -1
	
	return(latitude, longitude)
	

def copyGPSData(filenamein, filenameout):
	'''
	Reads the GPS and date/time information from an input TIFF file
	and copies that information to the resulting 5-band combined TIFF
	file
	
	Information is saved to the EXIF data of the output image
	'''
	#Read information from the input file
	metadatain = pyexiv2.ImageMetadata(filenamein)
	metadatain.read()
	print(metadatain.exif_keys)
	tagTime = metadatain["Exif.Image.DateTime"]
	tagLat = metadatain["Exif.GPSInfo.GPSLatitude"]
	tagLatRef = metadatain["Exif.GPSInfo.GPSLatitudeRef"]
	tagLong = metadatain["Exif.GPSInfo.GPSLongitude"]
	tagLongRef = metadatain["Exif.GPSInfo.GPSLongitudeRef"]
	
	#Read information from the output file
	#Write the GPS keys and values to the output file
	metadataout = pyexiv2.ImageMetadata(filenameout)
	metadataout.read()
	
	key = "Exif.Image.DateTime"
	value = tagTime.value 
	metadataout[key] = pyexiv2.ExifTag(key, value)
	
	key = "Exif.GPSInfo.GPSLatitude"
	value = tagLat.value
	metadataout[key] = pyexiv2.ExifTag(key, value)
	
	key = "Exif.GPSInfo.GPSLatitudeRef"
	value = tagLatRef.value 
	metadataout[key] = pyexiv2.ExifTag(key, value)
	
	key = "Exif.GPSInfo.GPSLongitude"
	value = tagLong.value 
	metadataout[key] = pyexiv2.ExifTag(key, value)
	
	key = "Exif.GPSInfo.GPSLongitudeRef"
	value = tagLongRef.value 
	metadataout[key] = pyexiv2.ExifTag(key, value)
	print(metadataout.exif_keys)
	
	return 0
	
def find_pictures(path_name, distance, originlat, originlng):
	
	'''
		Parses through a directory of .jpeg files and compares their
		GPS metadata to an input latitude and longitude
		The distance between the input latitude/longitude and the 
		coordinates of each picture is calculated.
		
		If the distance value is less than or equal to the input distance
		threshold, then the image is copied into the "Sorted directory
		
	'''
	
	files = glob.glob(path_name)
	#files = glob.glob(data_path)
	#source = path_name
	meta_data = []
	
	'''
		Grab the input path, split it, then add the Sorted folder to it
	'''
	tempsplit = path_name.split("*")
	print (tempsplit[0])
	distancedir = tempsplit[0] + "Sorted"
	deleteimages = os.listdir(distancedir)
	'''
		Create the Sorted folder if it does not exists
		Delete any existing .jpeg files if it does
	'''
	print(distancedir)
	if not os.path.exists(distancedir):
		os.makedirs(distancedir)
	
	
	if os.path.exists(distancedir):
		for file in deleteimages:
			if file.endswith(".jpg"):
				os.remove(os.path.join(distancedir, file))
				

	'''
		Parse through the origin folder of .jpeg files, calculate the
		distance between GPS coordinates, then copy the images who are
		under the input distance threshold into the Sorted folder
	'''
	
	for file in sorted(files):
		
		if (distance is not None):
			(lat, lng) = getGPSCoordinates(file)
	
			piclat = lat * 3.14159/180
			piclng = lng * 3.14159/180
			
			currlat = originlat * 3.14159/180
			currlng = originlng * 3.14159/180
		
			
			deltalat = piclat - currlat
			deltalng = piclng - currlng
			
			a = math.sin(deltalat/2) * math.sin(deltalat/2) + math.cos(currlat) * math.cos(piclat) * math.sin(deltalng/2) * math.sin(deltalng/2)
			c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
			d = 6371000 * c
			
			#print (a, c, d)
			if(d <= distance):
				print(d)
				#print(file)
				shutil.copy(file, distancedir)
		else:
			shutil.copy(file, distancedir)
		
		
	return 0
'''
	Faster, less accurate method of calculating ECC alignment between two images
	Much faster compared to ECCtransform function
	Also returns the "Iterations do not converge" for nearly every 5-band imageset
	
	The method works by calculating alignment between two small sections of the image
	If the alignment is correct, the algorithm uses the same calculation as a basis for a larger
	section of the two images. This process is repeated until the area of both images is covered
	and the second image alignment is calculated
'''

def pyramidECC(img1, img2, name, setNum, output_path):
	
	
	img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
	img2 = cv2.imread(img2, cv2.IMREAD_COLOR)
	
	init_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
	n_iters = 100
	e_thresh = 1e-6
	warp_mode = cv2.MOTION_EUCLIDEAN
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, e_thresh)
	h, w = img1.shape[:2]
	nol = 4
	warp = init_warp
	warp = warp * np.array([[1, 1, 2]], dtype = np.float32)**(1-nol)
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	gray1_pyr = [gray1]
	gray2_pyr = [gray2]
	
	for level in range(nol):
		gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx = 1/2, fy = 1/2, interpolation = cv2.INTER_AREA))
		
		gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx = 1/2, fy = 1/2, interpolation = cv2.INTER_AREA))
		
	#Run pyramid ECC 
	
	for level in range(nol):
		
		cc, warp = cv2.findTransformECC(get_gradient(gray1_pyr[level]), get_gradient(gray2_pyr[level]), warp, warp_mode, criteria)
		
		if level != nol-1:
			warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype = np.float32)
	
	
	img2_aligned = cv2.warpAffine(img2, warp, (w, h), flags=cv2.WARP_INVERSE_MAP)
	blended = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
	cv2.imwrite('pyr_blended.png', blended)	
	warp_diff = cv2.absdiff(img2_aligned, img1)
	cv2.imwrite('pyr_diff.png', warp_diff)
		
	cv2.imwrite((output_path + str(setNum) + "_" + str(name + 1) + "result5band.tif"),	img2_aligned)
	
	
	return 0		
	
'''
	Render the tif images in a given folder
	The rendering colormap uses the Red-Yellow-Green color values used
	in QGis 2.
	Red = -1
	Orange = -0.5
	Yellow = 0
	Yellow/Green = 0.5
	Green = 1
	
	Colormap scale used for comparing values used in NDVI and NDRE
'''
							
def render(path_name):
	
	
	result_path = path_name + "Rendered/"
	if not(os.path.isdir(result_path)):
		os.mkdir(result_path)
		
	deleteimages = glob.glob(result_path + "*.tif")
	
	if os.path.exists(result_path):
		for file in deleteimages:
			if file.endswith(".tif"):
				os.remove(os.path.join(result_path, file))
	
	textDir = "/home/roark/Documents/USRT Summer 2018/Demo Pics/"
	files = path_name + "*.tif"
	files = glob.glob(files)
	imageList = []
	for file in sorted(files):
		imageList.append(file)
	
	
	for i in range(len(sorted(imageList))):
		tempsplit = imageList[i].split(".")
		fileName = tempsplit[0]
		
		subprocess.call(["gdaldem", "color-relief", (fileName + ".tif"), (textDir + "RdYlGn.txt"), (fileName + "_render" + ".tif")])
		render_temp = fileName + "_render" + ".tif"
		print(render_temp)
		shutil.move(render_temp, result_path)
		
	
	return 0

'''
	Iteratively checks the image for values exceeding the recommended threshold
	Usually this indicates that the edges of the array are sticking out 
	and are approximately -0.99 or 0.99
	In this case, keep cropping the image by small increments until
	the edge cases no longer exist or the number of iterations equals 4
'''

def cropEdges(path_name):
	
	
	
	if not(os.path.isdir(path_name)):
		os.mkdir(path_name)
	'''
	deleteimages = glob.glob(path_name + "*.tif")
	
	if os.path.exists(path_name):
		for file in deleteimages:
			if file.endswith(".tif"):
				os.remove(os.path.join(path_name, file))
	'''	
	files = path_name + "*.tif"
	print(files)
	files = glob.glob(files)
	imageList = []
	for file in sorted(files):
		imageList.append(file)
	
	for i in range(len(sorted(imageList))):
		
		print(imageList[i])
		convert_tif = gdal.Open(imageList[i])
		#print(convert_tif)
		convert = convert_tif.ReadAsArray()
		#print(convert)
		
		
		decreaseFactor = 0.01
		iterations = 0
		crop_img = convert
		while(((np.amin(convert) < -0.95) or (np.amax(convert) > 0.95)) and iterations < 4):
			print(np.amin(convert))
			print(np.amax(convert))
			
			height = convert.shape[0]
			width = convert.shape[1]
			crop_img = convert[int(height * decreaseFactor):int(height * (1 - decreaseFactor)), int(width * decreaseFactor):int(width * (1 - decreaseFactor))]
			decreaseFactor = decreaseFactor  + 0.005
			convert = crop_img
			iterations = iterations + 1
		cv2.imwrite((imageList[i]), crop_img)
			
		
	return 0	
	

'''
	Function for rendering the NDVI/NDRE image in a Red/Yellow/Green color
	ramp. Unused for now, still testing.
'''

	
def get_gradient(image):
	#Calculate the x and y gradients using Sobel operator
	grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
	
	#Combine the two gradients
	grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
	return grad


def imAlign(img1, img2, name, setNum, output_path, iterations, corrCoeff):
	
	
	im1 = cv2.imread(img1)
	im2 = cv2.imread(img2)
	#im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
	#im2 = cv2.imread(img2, cv2.IMREAD_COLOR)

	#if(name == 3):
	#	cv2.imwrite((output_path + str(setNum) + "_" + str(name + 1) + "result5band.tif"), im2)
	
	#cv2.imshow(im1, "Rededge unaltered")
	
	print("All good")
	#print(im1, im2)
	im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	
	#im1_gray = im1
	#im2_gray = im2
	
	sz = im1.shape
	
	#Define the motion model
	#cv::MOTION_TRANSLATION = 0,
	#cv::MOTION_EUCLIDEAN = 1,
	#cv::MOTION_AFFINE = 2,
	#cv::MOTION_HOMOGRAPHY = 3 
	#warp_mode = cv2.MOTION_TRANSLATION
	warp_mode = cv2.MOTION_HOMOGRAPHY
	#Define 2x3 or 3x3 matrices and initialize the matrix to id
	if warp_mode == cv2.MOTION_HOMOGRAPHY:
		warp_matrix = np.eye(3, 3, dtype=np.float32)
	else:
		warp_matrix = np.eye(2, 3, dtype=np.float32)
	
	#Specify the number of iterations 
	#Usually 1000 in normal operations
	number_of_iterations = iterations
	
	#Specify the threshold of the increment
	#In the correlation coefficient between the two iterations
	#termination_eps = 1e-4;
	termination_eps = corrCoeff
	#Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC(get_gradient(im1_gray),get_gradient(im2_gray),warp_matrix, warp_mode, criteria)
	
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		# Use warpPerspective for Homography transformations, not necessary for others 
		im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else:
		# Use warpAffine for Translation, Euclidean and Affine
		im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	#If the image out of the imageset is the first blue image, then it will have RGB bands for whatever reason
	#then separate the bands and write the third, blue band as the new im2_aligned matrix.
	#Write the result.
	if((name+1) == 1):
		blue1, blue2, blue3 = cv2.split(im2_aligned)
		im2_aligned = blue3
		
	cv2.imwrite((output_path + "align_" + str(setNum) + "_" + str(name + 1) + ".tif"), im2_aligned)
	
	
 	
	return 0;

def align5bands(input_path, output_path):
	
	print("Gets here")
	
	
	#Run through the input path and pick up 
	dirs = os.listdir(input_path)

	if not os.path.exists(output_path):
		os.makedirs(output_path)
	
	imageList = []
	dirList = []
	
	#Read all .tif files in the directory
	for file in sorted(dirs):
		if(file.endswith(".tif")):
			#urlArr = file.name.split("_")
			#print(urlArr[0], urlArr[1])
			dirList.append(file)
	
	#Check to see if they follow the correct naming format
	for i in range(0,len(dirList)):
		print(dirList[i])
		urlArr = dirList[i].split("_")
		if((len(urlArr) > 1) & (urlArr[0] == "IMG")):
			imageList.append(dirList[i])
			print(urlArr[0], urlArr[1], urlArr[2])
			
	#Create the output path if it does not exists
	result_path = output_path
	if not(os.path.isdir(result_path)):
		os.mkdir(result_path)

	
	imNum = 0
	setNumber = 0
	print("Total Number of Images: ", len(imageList))
	
	if(len(imageList)%5 != 0):
		print("Incorrect number of images. Some imagesets may not be complete")
	'''
	Parse through a list of tif images
	For each directory
	'''
	
	while(imNum < len(imageList)):
		
		#Initializes image order number
		order = 0
		
		#Creates imageset value 
		urlArr = []
		img = []
		urlArr = imageList[imNum].split("_")
		tempNum = urlArr[1]
		for i in range(imNum, imNum+5):
			j = (i+1) - imNum
			urlArr = imageList[i].split("_")
			if(tempNum == urlArr[1]):
				url = input_path + "IMG_" + str(tempNum) + "_" + str(j) + ".tif"
				img.append(url)	

		for i in range(0, 5):
			print(i, img[i])
		iterations = 5000
		corrCoeff = (1e-10)
		
		'''
			Align all five bands by the near infrared band using the ECCTransfom algorithm
			If it gives the "Iterations do not converge" error, align the three RGB bands together
			Then merge NIR and Red Edge bands to Green and Blue bands separately.
			Align the 3-band NIR and Red Edge bands to the RGB picture, then separate
			the NIR and Red Edge channels into their own images
		'''
		try:
			for i in range(0, 5):
				imAlign(img[3], img[i], i, tempNum, output_path, iterations, corrCoeff)
				#pyramidECC(img[3], img[i], i, tempNum, output_path)
		except cv2.error as e:

			for i in range(0, 3):
				imAlign(img[2], img[i], i, tempNum, output_path, iterations, corrCoeff)
			
			
			
				
			subprocess.call(["gdalbuildvrt", "-separate", "RGB.vrt", ("align_" + str(tempNum) +"_1.tif"), ("align_" + str(tempNum) +"_2.tif"), ("align_" + str(tempNum) +"_3.tif")], cwd = output_path)
			
			#subprocess.call(["gdalbuildvrt", "-separate", "IRGB.vrt", (str(img[3])), ("align_" + str(tempNum) + "_2.tif"), ("align_" + str(tempNum) + "_1.tif")], cwd = output_path) 
			subprocess.call(["gdalbuildvrt", "-separate", "IRGB.vrt", (str(img[3])), (str(img[2])), (str(img[1]))], cwd = output_path) 
			
			#subprocess.call(["gdalbuildvrt", "-separate", "REGB.vrt", (str(img[4])), ("align_" + str(tempNum) + "_2.tif"), ("align_" + str(tempNum) + "_1.tif")], cwd = output_path)
			subprocess.call(["gdalbuildvrt", "-separate", "REGB.vrt", (str(img[4])), (str(img[2])), (str(img[1]))], cwd = output_path)
			
			subprocess.call(["gdal_translate", "RGB.vrt", "RGB.tif"], cwd = output_path) 
			subprocess.call(["gdal_translate", "IRGB.vrt", "IRGB.tif"], cwd = output_path)
			subprocess.call(["gdal_translate", "REGB.vrt", "REGB.tif"], cwd = output_path)
			
			RGB = output_path + "RGB.tif"
			IRGB = output_path + "IRGB.tif"
			REGB = output_path + "REGB.tif"
			
			print(RGB)
			print(IRGB)
			print(REGB)
			imAlign(RGB, img[3], 3, tempNum, output_path, iterations, corrCoeff)
			imAlign(RGB, img[4], 4, tempNum, output_path, iterations, corrCoeff)
			
			IRsep = cv2.imread(output_path + "align_" + str(tempNum) + "_4.tif", cv2.IMREAD_UNCHANGED)
			REsep = cv2.imread(output_path + "align_" + str(tempNum) + "_5.tif", cv2.IMREAD_UNCHANGED)
			
			print(output_path + "align_" + str(tempNum) + "_4.tif")
			#print(IRsep.size)
			
			IR1, IR2, IR3, = cv2.split(IRsep)
			RE1, RE2, RE3 = cv2.split(REsep)
			
			print(IR1)
			print(RE1)
			cv2.imwrite("align_" + str(tempNum) + "_4.tif", IR3)
			cv2.imwrite("align_" + str(tempNum) + "_5.tif", RE3) 
			
			tif = str(output_path + "RGB.tif")
			vrt = str(output_path + "RGB.vrt")
			if (os.path.exists(tif)):
				os.remove(tif)
			if (os.path.exists(vrt)):
				os.remove(vrt)
			tif = str(output_path + "IRGB.tif")
			vrt = str(output_path + "IRGB.vrt")
			if (os.path.exists(tif)):
				os.remove(tif)
			if (os.path.exists(vrt)):
				os.remove(vrt)
			tif = str(output_path + "REGB.tif")
			vrt = str(output_path + "REGB.vrt")
			if (os.path.exists(tif)):
				os.remove(tif)
			if (os.path.exists(vrt)):
				os.remove(vrt)
			
			
			
				
		imNum = imNum + 5
		
'''
	Reads a five-band input TIFF file created from a virtual raster
	and processes them based on agricultural algorithms
	Bands are ordered left to right
	1 - Blue
	2 - Green
	3 - Red
	4 - Red Edge
	5 - NIR

	Read the third and fifth bands to make an NDVI Image
	(NIR - RED) / (NIR + RED)

	Grabs the fourth and fifth bands and creates the NDRE image
	(NIR - REDGE) / (NIR + REDGE)
	
	Grabs the second and fifth bands and creates the GNDVI image
	(NIR - GREEN) / (NIR + GREEN)

	Grabs the SECOND and fifth bands and creates the GNDVI image
	(NIR)/(Green)

'''


# specify order of bands in input raster
blueband_num = 1
greenband_num = 2
redband_num = 3
NIRband_num = 4
rededgeband_num = 5

'''
Generates a combined 5-band tif file for each set of appropriately
labeled imagesets in the given directory. 
The function loads in the list of images that match the required naming
types and calls the GDAL functions from the terminal command line
The five tiff files are converted into a virtual raster, and that virtual
raster is converted into a 5-band tiff file that can be used to calculate
agricultural data arrays
'''

def generateTiffs(path_name):

		result_path = path_name + "results/"
		if not(os.path.isdir(result_path)):
			os.mkdir(result_path)
		
		files = path_name + "align*.tif"
		files = glob.glob(files)
		imageList = []
		for file in sorted(files):
			imageList.append(file)
		
		i = 0
		while(i < len(imageList)):
			print(imageList[i])
			i = i+1
		
		i = 0
		numFile = -1
		
		for i in range(len(sorted(imageList))):
			tempsplit = imageList[i].split("_")
			
			if int(numFile) != int(tempsplit[1]):
				numFile = tempsplit[1]
				fileDec = tempsplit[0]
				subprocess.call(["gdalbuildvrt", "-separate", (numFile + ".vrt"), (str(fileDec) + "_" + numFile + "_1.tif"), (str(fileDec) + "_" + numFile + "_2.tif"), (str(fileDec) + "_" + numFile + "_3.tif"), (str(fileDec) + "_" + numFile + "_4.tif"), (str(fileDec) + "_" + numFile + "_5.tif")], cwd = path_name)
				subprocess.call(["gdal_translate", (numFile + ".vrt"), ("results/" + numFile + ".tif")], cwd = path_name)
		numList = len(imageList)/5		
			

		return 0

'''
	Takes all appropriately named 5-band tiffs and makes the given 
	calculation for each 5-band tiff. The resulting image is saved to the
	output path in its own appropriately named directory 
'''

def calculateAlgorithm(input_path, calculation):
	


	result_path = input_path + str(calculation) + "/"
	print(result_path)
	if not(os.path.isdir(result_path)):
		os.mkdir(result_path)

	inroot = input_path
	outroot = result_path
	
	print(inroot)
	print(outroot)
	
	blueband_num = 1
	greenband_num = 2
	redband_num = 3
	NIRband_num = 4
	rededgeband_num = 5
	
	
	files = glob.glob(input_path + "*.tif")
	imageList = []
	for file in sorted(files):
		if file.endswith(".tif"):
			imageList.append(file)
	
	np.seterr(invalid='ignore')
	for i in range(0, len(imageList)):
		# input raster
		tempSplit = imageList[i].split(".")
		inputRaster_path = os.path.join(inroot, imageList[i])
		outputRaster_path = os.path.join(outroot, tempSplit[0] + str(calculation) + ".tif")
		
		# read input rows, cols, and bands of raster
		ds = gdal.Open(inputRaster_path, GA_ReadOnly)
		nrows = ds.RasterYSize
		ncols = ds.RasterXSize
		nbands = ds.RasterCount
		
		# read as raster 
		blueband_raster = ds.GetRasterBand(blueband_num)
		greenband_raster = ds.GetRasterBand(greenband_num)
		redband_raster = ds.GetRasterBand(redband_num)
		nirband_raster = ds.GetRasterBand(NIRband_num)
		rededgeband_raster = ds.GetRasterBand(rededgeband_num)
		
		
		# set up block size as 500 pixels, you can actually set an larger number if your ram is big enough
		block_size = 500
		
		# set up output parameters
		format = "Gtiff"  
		driver = gdal.GetDriverByName(format)  
		dst_ds = driver.Create(outputRaster_path, ncols, nrows, 1, gdal.GDT_Float32)  
		dst_ds.SetGeoTransform(ds.GetGeoTransform())  
		dst_ds.SetProjection(ds.GetProjection())  
		
		# segment rows and cols
		ysize = nrows
		xsize = ncols
		
		# read row
		for i in range(0, ysize, block_size):
		
			# don't want moving window to be larger than row size of input raster
			if i + block_size < ysize:  
				rows = block_size  
			else:  
				rows = ysize - i
			
			# read col      
		
			for j in range(0, xsize, block_size):
		
				# don't want moving window to be larger than col size of input raster
				if j + block_size < xsize:  
					cols = block_size  
				else:  
					cols = xsize - j 
				
				# get block out of the whole raster
				nir_array = nirband_raster.ReadAsArray(j, i, cols, rows)
				if calculation == "NDVI":
					red_array = redband_raster.ReadAsArray(j, i, cols, rows)
					numerator = (nir_array + 0.001) - red_array 
					denominator = (nir_array + 0.001) + red_array

				elif calculation == "NDRE":
					rededge_array = rededgeband_raster.ReadAsArray(j, i, cols, rows)
					numerator = (nir_array + 0.001) - rededge_array
					denominator = (nir_array + 0.001) + rededge_array

				elif calculation == "GNDVI":
					green_array = greenband_raster.ReadAsArray(j, i, cols, rows)
					numerator = (nir_array + 0.001) - green_array
					denominator = (nir_array + 0.001) + green_array

				elif calculation == "ENDVI":
					green_array = greenband_raster.ReadAsArray(j, i, cols, rows)
					blue_array = blueband_raster.ReadAsArray(j, i, cols, rows)
					numerator = (((nir_array + 0.001) + green_array) - (2 * blue_array))
					denominator = (((nir_array + 0.001) + green_array) + (2 * blue_array))

				elif calculation == "GRVI":
					green_array = greenband_raster.ReadAsArray(j, i, cols, rows)
					numerator = (nir_array + 0.001)
					denominator = green_array

				else:
					print("Invalid formula label.")
					
					
				result = numerator / denominator 
			
		        
				# write ndvi array to tiff file
				dst_ds.GetRasterBand(1).WriteArray(result, j, i) 
	 
		file = tempSplit[0] + calculation + ".tif"
		shutil.move(file, result_path)
	
	cropEdges(result_path)
	render(result_path)
		
	# ends program
	dst_ds = None 
	print ('Program ends')
	return 0
	
'''
	Moves all images into their appropriate folders based on the name of
	the tiff file
'''
def moveImages(input_path, output_path):
	files = glob.glob(input_path + "*.tif")
	print(input_path)
	print(output_path)
	for file in sorted(files):
		print (file)
		if file.endswith("NDVI.tif"):
			print (file)
			shutil.move(file, output_path + "NDVI/")
		if file.endswith("NDRE.tif"):
			print (file)
			shutil.move(file, output_path + "NDRE/")
		if file.endswith("GNDVI.tif"):
			print (file)
			shutil.move(file, output_path + "GNDVI/")
	return 0

def delete(files):
	
	files = glob.glob(files)
	
	tempList = []
	for file in files:
		tempList.append(file)
	
	for i in range(0, len(tempList)):
		if(os.path.exists(tempList[i])):
			os.remove(tempList[i])

def delete_images(output_path, result_path):
	
	
	#files = output_path + "*.vrt"
	#delete(files)
	
	#files = output_path + "*.tif"
	#delete(files)
	
	im_list = []
	
	im_list.append(result_path + "NDVI/")
	im_list.append(result_path + "NDRE/")
	im_list.append(result_path + "GRVI/")
	im_list.append(result_path + "GNDVI/")
	im_list.append(result_path + "ENDVI/")
	
	for i in range(0, len(im_list)):
		if (os.path.exists(im_list[i])):
			shutil.rmtree(im_list[i])
	return
	
	
def initialize(input_path, output_path, rerun):
	
	result_path = output_path + "results/"
	
	if(rerun == True):
		delete_images(output_path, result_path)
	
	if not(os.path.isdir(result_path)):
		os.mkdir(result_path)
		
	align5bands(input_path, output_path)
	generateTiffs(output_path)
	
	calculateAlgorithm(result_path, "NDVI")
	calculateAlgorithm(result_path, "NDRE")
	calculateAlgorithm(result_path, "GNDVI")
	calculateAlgorithm(result_path, "GRVI")
	calculateAlgorithm(result_path, "ENDVI")
	return

def main():
	
	#input_path = "/home/roark/Pictures/jpegtest/distresults/Fourth/000/Sorted/"
	#output_path = "/home/roark/Pictures/Align/8-13-2018 MergeComp/"
	input_path = "/media/roark/CONOR'S1/Fourth Set (Closest to Ground)/Stitched/Willow/"
	output_path = "/home/roark/Pictures/DDemo/"
	rerun = False
	initialize(input_path, output_path, rerun)



	return 0

if __name__ == '__main__':
	main()


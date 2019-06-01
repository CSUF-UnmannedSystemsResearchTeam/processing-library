#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#


from datetime import datetime, timedelta
import pytz
import os
import math
import pyexiv2

import PIL
from PIL.ExifTags import TAGS, GPSTAGS
from osgeo import gdal

class Metadata(object):
	
	'''
	Container for Micasense image metadata
	'''
	def __init__(self, filename):
			
		self.metadata = pyexiv2.ImageMetadata(filename)
		self.metadata.read()
		
	def get_all(self):
		'''
		Get the metadata dictionary items
		'''
		return self.metadata
	
	def get_item(self, item, index = None):
		'''
		Get metadata item by Namespace.Type.Parameter
		'''
		val = None
		try:
			val = self.metadata[item].value
			if index is not None:
				#val = val[index].value
				val = val[index]
		except KeyError:
			#print ("Item " +item"  not found")
			pass
		except IndexError:
			print("Item is out of index range")
		return val
	
	def size(self, item):
		'''
		Get the length of the metadata item
		'''
		val = self.get_item(item)
		return len(val)
	
	def print_all(self):
		'''
			Prints out all items in the Metadata dictionary
		'''
		for item in self.metadata.exif_keys:
			try:
				print(item, self.get_item(item))
			except:
				print(item)
		for item in self.metadata.xmp_keys:
			try:
				print(item, self.get_item(item))
			except:
				print(item)
			#print(item)
		#for item in self.get_all():
		#	print((item, self.get_item(item)))
	
	def dls_present(self):
		return self.get_item("Xmp.Camera.Irradiance") is not None
		
	def supports_radiometric_calibration(self):
		if(self.get_item('Xmp.Camera.RadiometricCalibration')) is None:
			return False
		return True
	
	def position(self):
		
		'''
			Gets and calculates the WGS-84 latitude, longitude tuple as signed decimal degrees
		'''
		
		lat = self.get_item('Exif.GPSInfo.GPSLatitude')
		latref = self.get_item('Exif.GPSInfo.GPSLatitudeRef')
		latitude = float(lat[0]) + float(lat[1]) + float(lat[2])
		if latref == 'S':
			latitude *= -1.0
		
		lon = self.get_item('Exif.GPSInfo.GPSLongitude')
		lonref = self.get_item('Exif.GPSInfo.GPSLongitudeRef')
		longitude = float(lon[0]) + float(lon[1]) + float(lon[2])
		if lonref == 'W':
			longitude *= -1.0
		alt = float(self.get_item('Exif.GPSInfo.GPSAltitude'))
		return latitude, longitude, alt
	
	def utc_time(self):
		'''
		Get the timezone aware date-time of the image
		'''
		str_time = str(self.get_item('Exif.Photo.DateTimeOriginal'))
		utc_time = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
		subsec = int(self.get_item('Exif.Photo.SubSecTime'))
		negative = 1.0
		
		if subsec < 0:
			negative = -1.0
			subsec *= -1.0
		subsec = float('0.{}'.format(int(subsec)))
		subsec *= negative
		ms = subsec * 1e3
		#utc_time += timedelta(milliseconds = ms)
		#timezone = pytz.timezone('UTC')
		#utc_time = timezone.localize(utc_time)
		
		return utc_time
		
	def dls_pose(self):
		'''
		Get DLS pose as local earth-fixed yaw, pitch, roll in radians
		'''
		
		yaw = float(self.get_item('Xmp.DLS.Yaw'))
		pitch = float(self.get_item('Xmp.DLS.Pitch'))
		roll = float(self.get_item('Xmp.DLS.Roll'))
		
		return yaw, pitch, roll
		
	def dls_irradiance(self):
		return float(self.get_item('Xmp.DLS.SpectralIrradiance'))
		
	def capture_id(self):
		return self.get_item('Xmp.MicaSense.CaptureId')
		
	def flight_id(self):
		return self.get_item('Xmp.MicaSense.FlightId')
		
	def camera_make(self):
		return self.get_item('Exif.Image.Make')
	
	def camera_model(self):
		return self.get_item('Exif.Image.Model')
		
	def firmware_version(self):
		return self.get_item('Exif.Image.Software')
	
	def band_name(self):
		return self.get_item('Xmp.Camera.BandName')
		
	def band_index(self):
		return self.get_item('Xmp.Camera.RigCameraIndex')
	
	def exposure(self):
		exp = self.get_item('Exif.Photo.ExposureTime')
		if math.fabs(exp-(1.0/6329.0)) < 0.0001:
			exp = 0.000274
		return exp
	
	def gain(self):
		return self.get_item('Exif.Photo.ISOSpeed')
	
	def image_size(self):
		return self.get_item('Exif.Image.ImageWidth'), self.get_item('Exif.Image.ImageLength')
		
	def center_wavelength(self):
		return self.get_item('Xmp.Camera.CentralWavelength')
	
	def bandwidth(self):
		return self.get_item('Xmp.Camera.WavelengthFWHM')
		
	def radiometric_cal(self):
		nelem = self.size('Xmp.MicaSense.RadiometricCalibration')
		val = self.get_item('Xmp.MicaSense.RadiometricCalibration')
		floatVal = []
		
		for i in val:
			floatVal.append(float(i))
		#radList = []
		#for i in range(nelem):
		#	radList.append(float(self.get_item('Xmp.MicaSense.RadiometricCalbration', i)))
		return floatVal
		#return [float(self.get_item('Xmp.MicaSense.RadiometricCalibration', i)) for i in range(nelem)]
		#return radList
		
	def black_level(self):
		black_lvl = self.get_item('Exif.Image.BlackLevel')
		total = 0.0
		num = len(black_lvl)
		for pixel in black_lvl:
			total += float(pixel)
		return total/float(num)
		
	def dark_pixels(self):
		'''
			Get the average of the optically covered pixel values
			These pixels are raw and have not been radiometrically corrected.
			Use the black_level method for all radiometric calibrations
		'''
		dark_pixels = self.get_item('Xmp.MicaSense.DarkRowValue')
		total = 0.0
		num = len(dark_pixels)
		for pixel in dark_pixels:
			total += float(pixel)
		return total/float(num)
		
	def bits_per_pixel(self):
		'''
			Get the number of bits per pixel, which defines the maximum digital number value in the image
		'''
		
		return self.get_item('Exif.Image.BitsPerSample')

	def vignette_center(self):
		'''
			Get the vignette center in X and Y image coordinates
		'''
		val = self.get_item('Xmp.Camera.VignettingCenter')
		floatVal = []
		
		for i in val:
			floatVal.append(float(i))
		return floatVal
	
	def vignette_polynomial(self):
		'''
			Get the radial vignette polynomial in the order it's defined
			in the metadata
		'''
		val = self.get_item('Xmp.Camera.VignettingPolynomial')
		floatVal = []
		
		for i in val:
			floatVal.append(float(i))
		return floatVal
	
	def distortion_parameters(self):
		
		val = self.get_item('Xmp.Camera.PerspectiveDistortion')
		floatVal = []
		
		for i in val:
			floatVal.append(float(i))
		return floatVal
		
	def principal_point(self):
		
		val = self.get_item('Xmp.Camera.PrincipalPoint').split(",")
		floatVal = []

		for item in val:
			floatVal.append(float(item))
		return floatVal

	def focal_plane_resolution_px_per_mm(self):
		
		fp_x_resolution = float(self.get_item('Exif.Photo.FocalPlaneXResolution'))
		fp_y_resolution = float(self.get_item('Exif.Photo.FocalPlaneYResolution'))
		
		return fp_x_resolution, fp_y_resolution

	def focal_length_mm(self):
		units = self.get_item('Xmp.Camera.PerspectiveFocalLengthUnits')
		focal_length_mm = 0.0
		if units == 'mm':
			focal_length_mm = float(self.get_item('Xmp.Camera.PerspectiveFocalLength'))
		else:
			focal_length_px = float(self.get_item('Xmp.Camera.PerspectiveFocalLength'))
			focal_length_mm = focal_length_px / self.focal_plane_resolution_px_per_mm()[0]
			
		return focal_length_mm

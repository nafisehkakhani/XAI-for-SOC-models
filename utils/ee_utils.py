
import os
import ee
ee.Initialize()
import geemap
import math
from typing import List
import json
import requests
from geemap import geojson_to_ee, ee_to_geojson
from ipyleaflet import GeoJSON

collection_dict = {
                           'l8': "LANDSAT/LC08/C01/T1_TOA",
                           'hls':  "NASA/HLS/HLSL30/v002",
    }
  
# Band combinations for each sensor corresponding to final selected corresponding bands                        
sensor_band_dict = ee.Dictionary({'l8'  : ee.List([1,2,3,4,5,6]),
                                  'hls' : ee.List([1,2,3,4,5,6]),
    })

possibleSensors = ['hls'] # Specify which sensors 
cloud_cov = 20 #Specify cloud coverage for hls data

  
# band names
bandNames = ee.List(['blue','green','red','nir','swir1','swir2'])
STD_NAMES = ['blue','green','red','nir','swir1','swir2']
bandNumbers = [0,1,2,3,4,5]


# Basic shadow masking using sum of specified bands
# Tends to include hill shadows and water
shadowThresh = 0.1
shadowSumBands = ['nir','swir1','swir2']

def maskShadows(img):
      ss = img.select(shadowSumBands).reduce(ee.Reducer.sum())
      return img.mask(img.mask().And(ss.gt(shadowThresh)))
    

# Function to mask clouds, ensure data exists in every band, and defringe images
# Assumes the image is a Landsat image

def maskCloudsAndSuch(img: ee.Image, cloudThresh=20):
    
    global possibleSensors

    if 'l8' in possibleSensors:
       # Bust clouds
       cs = ee.Algorithms.Landsat.simpleCloudScore(img).select(['cloud']).gt(cloudThresh)

       # Make sure all or no bands have data
       numberBandsHaveData = img.mask().reduce(ee.Reducer.sum())
       allOrNoBandsHaveData = numberBandsHaveData.eq(0).Or(numberBandsHaveData.gte(7))

       # If it's Landsat 5- defringe by nibbling away at the fringes
       # allBandsHaveData = allOrNoBandsHaveData.focal_min(1, 'square', 'pixels', 8)

       # Make sure no band is just under zero
       allBandsGT = img.reduce(ee.Reducer.min()).gt(-0.001)

       # Combine masks
       final_mask = img.mask().And(cs.Not()).And(allOrNoBandsHaveData).And(allBandsGT)

       out = img.mask(final_mask)
    else: 
      out = img
      
    return out



# Function to handle empty collections that will cause subsequent processes to fail
# If the collection is empty, will fill it with an empty image
def fillEmptyCollections(inCollection: ee.ImageCollection, dummyImage: ee.Image) -> ee.ImageCollection:


    """
    Description:
    This function handles empty Earth Engine collections that might otherwise cause 
    subsequent processes to fail. It checks if the input collection contains any images. 
    If the collection is empty, it creates a new collection by adding an empty image (dummy image) to it. 
    This ensures that the collection is not empty and can be further processed without causing errors.

    Parameters:

     inCollection: The input Earth Engine image collection that needs to be checked and possibly filled.
     dummyImage: An Earth Engine image that serves as a placeholder or dummy image to fill empty collections.
    
    Returns:  An Earth Engine image collection. If the input collection is not empty, 
    it returns the original input collection. If the input collection is empty, it returns a new collection containing the dummy image.
    """   

    dummyCollection = ee.ImageCollection([dummyImage.mask(ee.Image(0))]);
    imageCount = inCollection.toList(1).length()
    return ee.ImageCollection(ee.Algorithms.If(imageCount.gt(0),inCollection,dummyCollection))


  #Helper function to get images from a specified sensor
def getCollection(sensor,startDate,endDate,startJulian,endJulian, maskShadows, fB):
        collectionName = collection_dict[sensor]
    
        #Start with an un-date-confined collection of iamges
        WOD = ee.ImageCollection(collectionName).filterBounds(fB)
            
        #Pop off an image to serve as a template if there are no images in the date range
        dummy = ee.Image(WOD.first())
    
        #Filter by the dates
        ls = WOD.filterDate(startDate,endDate).filter(ee.Filter.calendarRange(startJulian,endJulian))
    
        #Fill the collection if it's empty
        ls = fillEmptyCollections(ls,dummy);
    
        #Clean the collection up- clouds, fringes....
        ls = ls.map(maskCloudsAndSuch).select(sensor_band_dict.get(sensor),bandNames).map(maskShadows)

        return ls
  

def getImage(year, compositingPeriod, startJulian, endJulian, reducer, fB, crs):
    
    global cloud_cov
    
    # Define dates
    # y1Image = year
    # y2Image = year + compositingPeriod
    
    #var roiName = roiName;
    startDate = ee.Date.fromYMD(ee.Number(year),1,1).advance(startJulian,'day')
    endDate = ee.Date.fromYMD(ee.Number(year).add(ee.Number(compositingPeriod)),1,1).advance(endJulian,'day')
    # print('Acquiring composite for',startDate,endDate)


    #Get the images for composite and shadow model
    if 'hls' in possibleSensors:
      ls = getCollection('hls',startDate,endDate,startJulian,endJulian, maskShadows, fB).filter(ee.Filter.lt('CLOUD_COVERAGE', cloud_cov))
    
    # else: 
    #   ls = getCollection('hls',ee.Date('1000-01-01'),ee.Date('1001-01-01'),0,365, maskShadows, fB)
    
    elif 'l8' in possibleSensors:
      ls = getCollection('l8',startDate,endDate,startJulian,endJulian, maskShadows, fB)
    
    # else: 
    #   ls = getCollection('l8',ee.Date('1000-01-01'),ee.Date('1001-01-01'),0,365, maskShadows, fB)
    
    
    # ls = ee.ImageCollection(hls.merge(l8))
    composite = ls.reduce(reducer).select(bandNumbers,bandNames)
    composite = composite.mask(composite.mask().clip(fB))
    
    
    # Set up our final composite with bands we'd like to include
    composite = composite.select(['blue','green','red','nir','swir1','swir2']).multiply(10000).int16().clip(fB)
  
    ###################################################################################
    # Calculate visual params using min and max of image
    
    
    # if viz == True:
    #   # Create a Map object
    #   Map = geemap.Map() 

    #   # Get descriptive name
    #   # fullName = roiName + '_' + str(y1Image) + '_' + str(y2Image) + '_' + str(startJulian) + '_' + str(endJulian) + '_Composite'
    #   fullName = str(y1Image) + '_' + str(y2Image) + '_' + str(startJulian) + '_' + str(endJulian) + '_Composite'


    #   # # Perform reduction to get the minimum values for each band
    #   # viz_min = composite.reduceRegion(
    #   #  reducer=ee.Reducer.min(),
    #   #  scale=10,
    #   #  crs=crs,
    #   #  bestEffort=True,
    #   #  tileScale=16
    #   # )

    #   # # Perform reduction to get the maximum values for each band
    #   # viz_max = composite.reduceRegion(
    #   # reducer=ee.Reducer.max(),
    #   # scale=10,
    #   # crs=crs,
    #   # bestEffort=True,
    #   # tileScale=16
    #   # )

    #   # vizParams = {'min': 0.5, 'max': [viz_max.getNumber('red').getInfo(), viz_max.getNumber('green').getInfo(), 
    #   #                                  viz_max.getNumber('blue').getInfo()],'bands': ['red', 'green', 'blue'], 'gamma': 1.6}
       

    #   # # Display as True color composite
    #   # Map.addLayer(composite, vizParams, fullName)

    #   Map.addLayer(composite, {'bands': ['red', 'green', 'blue'], 'min': 0, 'max': 4000}, fullName)


    #   # Display the map
    #   Map.setCenter(0, 0, 2)
    #   Map
    # else:
    #     pass
    
    #Print final composite
    # print ("Landsat Composite:", composite) 

    return composite



# Cloud masking function.
def fmask(image):
        # see https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Cirrus
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)

        # Apply the scaling factors to the appropriate bands.
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)

        # Replace the original bands with the scaled ones and apply the masks.
        return image.addBands(opticalBands, None, True).updateMask(qaMask)

# NDVI
def getNDVI(inImage):
        # Calculate NDVI = (nir - red) / (nir + red)
        # ndvi = inImage.select("SR_B5", "SR_B4").normalizedDifference().rename("ndvi");
        ndvi = inImage.select("nir", "red").normalizedDifference().rename("ndvi");
        # Add band to image
        outStack = inImage.addBands(ndvi);

        return outStack

# EVI
def getEVI(inImage):
         
   """
         
   Definition: Landsat Enhanced Vegetation Index (EVI) 
   is similar to Normalized Difference Vegetation Index 
   (NDVI) and can be used to quantify vegetation greenness. 
   However, EVI corrects for some atmospheric conditions and 
   canopy background noise and is more sensitive in areas with dense vegetation.

   Input(s): Landsat image (Raster)
   Ouput(s): EVI image (Raster)

   """
   
   evi = inImage.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
   'NIR' : inImage.select('nir'),
   'RED' : inImage.select('red'),
   'BLUE': inImage.select('blue')}).rename('evi')

   # Add band to image
   outStack = inImage.addBands(evi)
    
   return outStack 

def getGEMI(inImage):
  
  """
   GEMI = SpectralIndex(
            short_name="GEMI",
            long_name="Global Environment Monitoring Index",
            formula="((2.0*((N ** 2.0)-(R ** 2.0)) + 1.5*N + 0.5*R)/(N + R + 0.5))*(1.0 - 0.25*((2.0 * ((N ** 2.0) 
                    - (R ** 2)) + 1.5 * N + 0.5 * R)/(N + R + 0.5)))-((R - 0.125)/(1 - R))",
            reference="http://dx.doi.org/10.1007/bf00031911",
            application_domain="vegetation",
            date_of_addition="2021-04-07",
            contributor="https://github.com/davemlz",
        ),
  """

  GEMI = inImage.expression('((2.0*((N ** 2.0)-(R ** 2.0)) + 1.5*N + 0.5*R)/(N + R + 0.5))*(1.0 - 0.25*((2.0 * ((N ** 2.0) - (R ** 2)) + 1.5 * N + 0.5 * R)/(N + R + 0.5)))-((R - 0.125)/(1 - R))', {
   'N' : inImage.select('nir'),
   'R' : inImage.select('red')}).rename('gemi')
  
  # Add band to image
  outStack = inImage.addBands(GEMI)
    
  return outStack 


def getGLI(inImage):

    """
     GLI=SpectralIndex(
            short_name="GLI",
            long_name="Green Leaf Index",
            formula="(2.0 * G - R - B) / (2.0 * G + R + B)",
            reference="http://dx.doi.org/10.1080/10106040108542184",
            application_domain="vegetation",
            date_of_addition="2021-04-07",
            contributor="https://github.com/davemlz",
        ),
    """

    GLI = inImage.expression('(2.0 * G - R - B) / (2.0 * G + R + B)', {
      'B' : inImage.select('blue'),
      'R' : inImage.select('red'),
      'G' : inImage.select('green')}).rename('gli')
    
    # Add band to image
    outStack = inImage.addBands(GLI)
    
    return outStack 


def getBI(inImage):
   
   """
   BI=SpectralIndex(
            short_name="BI",
            long_name="Bare Soil Index",
            formula="((S1 + R) - (N + B))/((S1 + R) + (N + B))",
            reference="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.465.8749&rep=rep1&type=pdf",
            application_domain="soil",
            date_of_addition="2022-04-08",
            contributor="https://github.com/davemlz",
        ),
   """

   BI = inImage.expression('((S1 + R) - (N + B))/((S1 + R) + (N + B))', {
      'B' : inImage.select('blue'),
      'R' : inImage.select('red'),
      'N' : inImage.select('nir'),
      'S1' : inImage.select('swir1'),}).rename('bi')
    
   # Add band to image
   outStack = inImage.addBands(BI)
    
   return outStack 


# EVI
def getEVIandCloud(inImage):
         
   """
         
   Definition: Landsat Enhanced Vegetation Index (EVI) 
   is similar to Normalized Difference Vegetation Index 
   (NDVI) and can be used to quantify vegetation greenness. 
   However, EVI corrects for some atmospheric conditions and 
   canopy background noise and is more sensitive in areas with dense vegetation.

   Input(s): Landsat image (Raster)
   Ouput(s): EVI image (Raster)

   """
   qaMask = inImage.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
   evi = inImage.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
   'NIR' : inImage.select('SR_B5').multiply(0.0000275).add(-0.2),
   'RED' : inImage.select('SR_B4').multiply(0.0000275).add(-0.2),
   'BLUE': inImage.select('SR_B2').multiply(0.0000275).add(-0.2)}).rename('evi')

   # Add band to image
   outStack = inImage.addBands(evi).updateMask(qaMask);
    
   return outStack   
    

# SAVI
def getSAVIandCloud(inImage):
    # Calculate SAVI
    qaMask = inImage.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    
    savi = inImage.expression('((NIR-Red)/(NIR+Red+0.5))*1.5', {
          'NIR': inImage.select("SR_B5").multiply(0.0000275).add(-0.2),
          'Red': inImage.select("SR_B4").multiply(0.0000275).add(-0.2)}).rename('savi')
  
    # Add band to image
    outStack = inImage.addBands(savi).updateMask(qaMask);

    return outStack


# SAVI
def getSAVIandCloud(inImage):
    # Calculate SAVI
    qaMask = inImage.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    
    savi = inImage.expression('((NIR-Red)/(NIR+Red+0.5))*1.5', {
          'NIR': inImage.select("SR_B5").multiply(0.0000275).add(-0.2),
          'Red': inImage.select("SR_B4").multiply(0.0000275).add(-0.2)}).rename('savi')
  
    # Add band to image
    outStack = inImage.addBands(savi).updateMask(qaMask);

    return outStack


# Mineral indices
def getMineralIndices(inImage):
  
  #Clay Minerals = swir1 / swir2
  clayIndex = inImage.select('swir1').divide(inImage.select('swir2')).rename('clayIndex')
  
  #Ferrous Minerals = swir / nir
  ferrousIndex = inImage.select('swir1').divide(inImage.select('nir')).rename('ferrousIndex')
  
  #Carbonate Index = (red - green) / (red + green)
  carbonateIndex = inImage.normalizedDifference(['red','green']).rename('carbonateIndex')

  #Rock Outcrop Index = (swir1 - green) / (swir1 + green)
  rockOutcropIndex = inImage.normalizedDifference(['swir1','green']).rename('rockOutcropIndex')
  
  #Add bands
  outStack = inImage.addBands([clayIndex, ferrousIndex, carbonateIndex, rockOutcropIndex])

  return outStack

def addTopography():
    elevation = ee.Image("USGS/SRTMGL1_003")
    # print ("elev is:", elevation);
    # Calculate slope and aspect
    topo = ee.Algorithms.Terrain(elevation)
    
    # get % slope
    slopeDeg = topo.select(1)
    slopeRads = slopeDeg.multiply(math.pi).divide(ee.Number(180));
    slopeTan = slopeRads.tan();
    slopePCT = slopeTan.multiply(ee.Number(100)).rename('slopePCT');
    
    # Add 8-direction aspect
    aspect = topo.select('aspect');
    aspectRad = aspect.multiply(math.pi).divide(180);
    aspectSin = aspectRad.sin().rename('sin');
    # aspectSin = aspectSin.multiply(10000).int32()
    aspectCos = aspectRad.cos().rename('cos');
    # aspectCos = aspectCos.multiply(10000).int32()
    aspect_8 = (aspect.multiply(8).divide(360)).add(1).floor().uint8().rename('aspect_8');
    # Add 3 equally-spaced sun azimuth hillshades
    hill_1 = ee.Terrain.hillshade(elevation,30).rename('hill_1');
    hill_2 = ee.Terrain.hillshade(elevation,150).rename('hill_2');
    hill_3 = ee.Terrain.hillshade(elevation,270).rename('hill_3');
    
    # Add topography bands to image composite
    topo = topo.float()
    # topo = topo.clip(compositeArea)
    topo = topo.select('elevation')\
               .addBands(slopePCT).addBands(aspectSin).addBands(aspectCos)\
               .addBands(aspect_8).addBands(hill_1).addBands(hill_2).addBands(hill_3);
    # topo = topo.int16();
    
    return topo

import pandas as pd
def random_point_csv_generator(bounding_roi, num_points=1000, file_name="random_points", start_id=10000000, seed=0):
    """
    Generates a csv file with random points in a given bounding region of interest.
    Args:
        bounding_roi (ee.Geometry): Region of interest to generate random points in. 
        num_points (int): Number of random points to generate.
        file_name (str): Name of the csv file to save the random points in.
        start_id (int): The starting id of the random points. This is to make the point_id an 8 digit number, same as LUCAS dataset.
        
    Returns:
        df (pandas.DataFrame): A dataframe with the random points and their coordinates.
    """
    # Create a random set of points in the region of interest
    points = ee.FeatureCollection.randomPoints(region=bounding_roi, points=num_points, seed=seed).getInfo()
    
    df = pd.DataFrame(columns=['Point_ID', 'long', 'lat','OC'])

    # loop through the features of the randomPoints object to populate the dataframe
    for i, point in enumerate(points['features']):
        lon = point['geometry']['coordinates'][0]
        lat = point['geometry']['coordinates'][1]
        
        df.loc[i] = [i+start_id, lon, lat, 0] # the start_id is to make it a 8 digit number, same as LUCAS dataset.

    # convert point_id column to integer type
    df['Point_ID'] = df['Point_ID'].astype(int)
    
    # save the dataframe as a csv file
    df.to_csv(file_name + ".csv", index=True, index_label='OID_')

    return df   

def calculate_land_cover_percentage(image, values, roi,scale=30):
    """
    Calculates the percentage of pixels in the given image that have the specified values within the ROI.

    Args:
    image (ee.Image): The image to calculate the percentage for.
    values (list): A list of values to calculate the percentage for. Check https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100 for the values.
    scale (int, optional): The scale of the reduction. Defaults to 30. The larger the scale, the faster the calculation, but the less accurate it is.
    roi (ee.Geometry, optional): The region of interest. Defaults to roi.

    Returns:
    float: The percentage of pixels in the image that have the specified values within the ROI.
    """
    # calculate the total number of pixels
    total_pixels = image.select('Map').reduceRegion(reducer=ee.Reducer.count(), geometry=roi, scale=scale).get('Map')
    value_pixels = ee.Number(0)
    for value in values:   
        value_p =  ee.Number(image.select('Map').eq(value).reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=scale).get('Map'))
        value_pixels = value_pixels.add(value_p)
    
    # calculate the percentage of pixels with the given value
    percentage = value_pixels.divide(total_pixels).multiply(100)
    
    return percentage


def get_square_roi(lat, lon, roi_size = 1920, return_gee_object = False):
    """
    Returns a square region of interest (ROI) centered at the given latitude and longitude
    coordinates with the specified size. By default, the ROI is returned as a list of
    coordinate pairs (longitude, latitude) that define the corners of the square. If
    `return_gee_object` is True, the ROI is returned as an Earth Engine geometry object.

    Args
    ----
        `lat` (float): Latitude coordinate of the center of the ROI.
        `lon` (float): Longitude coordinate of the center of the ROI.
        `roi_size` (int, optional): Size of the square ROI in meters. Default is 1920 meters. (about `64` pixels of `30m` resolution)
        `return_gee_object` (bool, optional): Whether to return the ROI as an Earth Engine geometry
            object instead of a list of coordinates. Default is False.

    Returns
    -------
        list or ee.Geometry.Polygon: If `return_gee_object` is False (default), a list of coordinate
            pairs (longitude, latitude) that define the corners of the square ROI. If `return_gee_object`
            is True, an Earth Engine geometry object representing the square ROI.

    Usage
    -----
        # Get a square ROI centered at lat=37.75, lon=-122.42 with a size of 1000 meters
        roi = get_square_roi(37.75, -122.42, roi_size=1000)
        print(roi)  # Output: [[-122.431, 37.758], [-122.408, 37.758], [-122.408, 37.741], [-122.431, 37.741], [-122.431, 37.758]]

    """

    # Convert the lat-long point to an EE geometry object
    point = ee.Geometry.Point(lon, lat)

    # Create a square buffer around the point with the given size
    roi = point.buffer(roi_size/2).bounds().getInfo()['coordinates']
    
    if return_gee_object:
        return ee.Geometry.Polygon(roi, None, False)
    else:
        # Return the square ROI as a list of coordinates
        return roi
# Satellite-Imagery-in-Economic-Nowcasting

Code from my work about applying satellite imagery in economic nowcasting. Preliminary results and a presentation at the Nationalbank of Ukraine can be found under this [Link](https://bank.gov.ua/en/research/events/279). The paper is currently under review and as soon as it's available I will post it here. The project started as my master thesis in Economics at the LMU Munich and was subsequently further developed by me after graduation.

tldr: I exploited very-high-resolution satellite imagery from the Pleiades and Pleiades Neo satellites to nowcast the economic development in Kyiv after the Russian invasion in February 2022. I did so by counting the number of vehicles (cars and trucks) in an area in the west of Kyiv. The change in the number of vehicles reflects the dramatic change in population following the full invasion. This is the main channel of influence for the change in economic activity. For further interpretation or results, the data is unfortunately too scarce. 

## Data Sources
[Airbus Pleiades Neo](https://space-solutions.airbus.com/imagery/our-optical-and-radar-satellite-imagery/pleiades-neo/)

[Airbus Pleiades](https://space-solutions.airbus.com/imagery/our-optical-and-radar-satellite-imagery/pleiades/)

[OpenStreetMap Ukraine](https://download.geofabrik.de/europe/ukraine-latest-free.shp.zip)

[GeoJSON Kyiv](https://osmtoday.com/europe/ukraine/kiev.html)

## Algorithm(s)
I used the YOLOv11 algorithm and trained it on annotated datasets that I made with Roboflow. I created datasets for images with 30cm, 50cm and 50cm resolution with snow and trained the algorithm specifically for these. Both offer Python packages (that you will find in the code) that take care of downloads etc.
All options I chose can be seen in the code.

## Pipeline
### I. Pre-Processing
The images are split into 640x640 chunks, the colors are rescaled and brightness and contrast are optimized. 
The roads are filtered by the intersection area of all images and certain road types are excluded.

### II. Training
The model is trained specifically on the annotated dataset. As the annotated dataset is stored in the Roboflow cloud only few commands from the Roboflow python package are required for this. After training the models are validated and deployed. The optimal confidence values are saved in a dictionary for later usage in prediction. 

### III. Prediction
The trained models are applied to the image chunks. All detected objects are stored in a gpd dataframe with box, confidence, label, image and coordinate. The coordinate is derived from the position of the detection box. 

### IV. Post-Processing
By calculating the shortes distance of a vehicle to the next road I can distinguish parking and moving cars approximately. Also I create an excel file with the number of vehicles on each image as well multiple images to display the results. 

## Results
![alt text](Intersection_Area.png "Intersection Area")

from fastapi import FastAPI, UploadFile, File, Request
from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server

from PIL import Image
import io
import numpy as np
from keras.models import load_model
from scipy import ndimage
import time
import uvicorn
import psutil

import warnings
warnings.filterwarnings('ignore')

import argparse



# Defining a function to parse command line arguments
def parse_command():

    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('path',type=str, help="Path of the model")
    
    ## parse the arguements
    args = parser.parse_args()

    return args.path


# Define a function to load a keras model stored on the local machine
def load_model_from_disk(file_path):
    model = load_model(file_path)
    return model

# Defining a function that takes in a serialized image and returns the predicted digit
def predict_digit(model, data_point):
    prediction = model.predict(data_point.reshape(1,-1))
    return str(np.argmax(prediction))

# Resizing the image to a 28x28 gray scale image
def format_image(image):
    img_gray = image.convert('L') # converting the image to grayscale
    img_resized = img_gray.resize((28, 28)) # resizing the image as 28x28 
    image_arr = np.array(img_resized) / 255.0 # scaling down the individual pixel values to the range (0,1)

    # Centering the image
    cy, cx = ndimage.center_of_mass(image_arr)
    rows, cols = image_arr.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = ndimage.shift(image_arr, (shifty, shiftx), cval=0)
    
    # Flatten the image and return it
    return img_centered.flatten()

# Defining a function to get memory allocation
def process_memory():
    return psutil.virtual_memory().used/(1024)



app = FastAPI(title='MNIST Application')

custom_registry = CollectorRegistry()

# Defining the necessary gauges and counters
api_run_time_gauge = Gauge("api_run_time_milliseconds", "API run time in milliseconds", registry=custom_registry)
api_tl_time_gauge = Gauge("api_tl_time", "API T/L time in microseconds per character", registry=custom_registry)

api_usage_counter = Counter("api_usage_counter", "Counts the number of API calls", ["client_ip"], registry=custom_registry)

api_memory_usage = Gauge('api_memory_usage', 'Memory usage of the API process', registry=custom_registry)
api_cpu_usage = Gauge('api_cpu_usage_percent', 'CPU usage of the API process', registry=custom_registry)

api_network_bytes_sent = Gauge('api_network_bytes_sent', 'Network bytes sent by the API process', registry=custom_registry)
api_network_bytes_received = Gauge('api_network_bytes_received', 'Network bytes received by the API process', registry=custom_registry)


# Metric instrumentation
Instrumentator().instrument(app).expose(app)

@app.post('/predict')
async def predict(file: UploadFile, request: Request):
    client_ip = request.client.host
    api_usage_counter.labels(client_ip).inc()

    start_time = time.time() # Start time for the API call
    memory_usage_start = process_memory() 
    network_io_counters = psutil.net_io_counters()

    # Reading the contents of the uploaded file
    contents = await file.read() 
    image = Image.open(io.BytesIO(contents)) 

    # Converting the image into the appropriate format
    image_arr = format_image(image)

    # Reading the command line argument that stores the path of the model
    file_path = parse_command()
    model = load_model_from_disk(file_path)
    
    prediction = predict_digit(model, image_arr)

    cpu_percent = psutil.cpu_percent(interval=1) 
    memory_usage_end = process_memory()      

    api_cpu_usage.set(cpu_percent)                                            
    api_memory_usage.set((np.abs(memory_usage_end-memory_usage_start)))   
    api_network_bytes_sent.set(network_io_counters.bytes_sent)             
    api_network_bytes_received.set(network_io_counters.bytes_recv)  

    end_time = time.time() # End time for API run time

    api_run_time = (end_time - start_time) * 1000  # in milliseconds
    api_run_time_gauge.set(api_run_time)

    # Calculate T/L time
    tl_time = (api_run_time / len(contents)) * 1000  # in microseconds per character
    api_tl_time_gauge.set(tl_time)

    # Return prediction along with API metrics
    return {"digit": prediction, "api_run_time": api_run_time, "api_tl_time": tl_time}


if __name__ == '__main__':
    start_http_server(8000)
    # Running the web-application defined earlier
    uvicorn.run(
        "CH20B065_assign7:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
    )

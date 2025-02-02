#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:39:06 2025

@author: toluojo
"""


'''
DEVELOPING A FAST API SERVER FOR INFERENCE 
'''

import os
import json
from fastapi import FastAPI, HTTPException, Request # USING FASTAPI to build API'S 
import torch
from pydantic import BaseModel
import numpy as np
from nltk.tokenize import word_tokenize
from data_processing import word_to_index, normalize_text, preprocess_text
from datetime import datetime
from fastapi.responses import FileResponse

'''
DEVELOPING A FAST API SERVER FOR INFERENCE 
'''


# Load your trained model
class UpvotePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(UpvotePredictor, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = torch.nn.Dropout(0.5)  # Add dropoutlayer to prevent overfitting and improve generalisation
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x

''' 
LOAD THE MODEL 
'''

vocab_size = 63642  # Replace with your actual vocabulary size
input_dim = 100  # Replace with your model's input dimension
hidden_dims = [128, 64, 32]  # Replace with your model's hidden dimensions
output_dim = 1  # Replace with your model's output dimension
model = UpvotePredictor(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load("cbow_model2.pth"))
model.eval()



# Model version
MODEL_VERSION = "0.1.0"

# Log directory
LOG_DIR_PATH = "/var/log/app"
LOG_PATH = f"{LOG_DIR_PATH}/V-{MODEL_VERSION}.log"

# Ensure the log directory exists
os.makedirs(LOG_DIR_PATH, exist_ok=True)


'''
DEFINE THE FASTAPI APP
'''
app = FastAPI()


'''
DEFINE REQUEST SCHEMA
'''
# THIS IS USED IN FAST API TO DEFINE AND VALIDATE REQUEST DADTA AUTOMATICALLY 
# THIS HELPS so when API receives data e.g. JSON Request it needs to check if data has correct structure and types 
# Define the request body schema
class PredictionRequest(BaseModel):
    title: str  # Ensures the request must have a "title" field as a string
    author: str
    timestamp: str 


# Serve the frontend
@app.get("/")
def home():
    return FileResponse("index.html")

# Health check endpoint
@app.get("/ping")
def ping():
    return "ok"

# Version endpoint
@app.get("/version")
def version():
    return {"version": MODEL_VERSION}

# Logs endpoint
@app.get("/logs")
def logs():
    return read_logs(LOG_PATH)


'''
PREDICTION ENDPOINT - 
'''

# register a function that handles POST request - send data to server - user submits an input 
@app.post("/predict") # this is a decorator in FAST API, Defines the endpoint handles POST Request 


def how_many_upvotes(post: PredictionRequest):
    start_time = datetime.utcnow().timestamp()  # Start time for latency
    prediction = predict_upvotes(post)  # Placeholder function
    end_time = datetime.utcnow().timestamp()  # End time
    latency = (end_time - start_time) * 1000  # Convert to ms

    message = {
        "Latency": latency,
        "Version": MODEL_VERSION,
        "Timestamp": end_time,
        "Input": post.dict(),
        "Prediction": prediction,
    }

    log_request(LOG_PATH, message)  # Save log
    return {"upvotes": prediction}



# PREDICTION FUNCTION 
def predict_upvotes(request: PredictionRequest):
    try:
        # Preprocess the input title
        title = request.title
        normalized_title = normalize_text(title)
        processed_tokens = preprocess_text(word_tokenize(normalized_title))
        token_embeddings = tokens_to_embeddings(processed_tokens, model, word_to_index)
        average_embedding = average_pooling(token_embeddings)

        if average_embedding is None:
            raise HTTPException(status_code=400, detail="No valid embeddings found for the title.")

        # Convert to tensor and make prediction
        average_embedding_tensor = torch.tensor(average_embedding).float().unsqueeze(0)
        predicted_score = model(average_embedding_tensor).item()

        return {"title": title, "predicted_score": predicted_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def tokens_to_embeddings(tokens, model, word_to_index):
    embeddings = []
    for token in tokens:
        if token in word_to_index:
            token_index = word_to_index[token]
            token_embedding = model.embeddings(torch.tensor(token_index)).detach().numpy()
            embeddings.append(token_embedding)
    return embeddings

def average_pooling(embeddings):
    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)


'''
NEW
'''

# Logging function
def log_request(log_path, message):
    with open(log_path, "a") as f:
        f.write(json.dumps(message) + "\n")

# Read logs function
def read_logs(log_path):
    try:
        with open(log_path, "r") as f:
            return {"logs": f.readlines()}
    except FileNotFoundError:
        return {"logs": []}



'''
THIS STARTS THE FAST API SERVER 
'''


# Run the server
if __name__ == "__main__": # Ensures the script runs only when executed directly, not when imported as a module.
    import uvicorn # imports a lightweight ASGI server used to run FastAPI applications
    uvicorn.run(app, host="0.0.0.0", port=8000) # Starts the FastAPI app (app) on host 0.0.0.0 (accessible from any device) and port 8000 (default FastAPI port)
    
    
    
    
    
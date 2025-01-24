from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum
import os


# Instantiate the app
app = FastAPI()

# Server our react application at the root
app.mount("/", StaticFiles(directory=os.path.join("frontend",
          "build"), html=True), name="build")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Permits requests from all origins.
    # Allows cookies and credentials to be included in the request.
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods.
    allow_headers=["*"]     # Allows all headers.
)


# Define the Lambda handler
handler = Mangum(app)


# Prevent Lambda showing errors in CloudWatch by handling warmup requests correctly
def lambda_handler(event, context):
    if "source" in event and event["source"] == "aws.events":
        print("This is a warm-ip invocation")
        return {}
    else:
        return handler(event, context)
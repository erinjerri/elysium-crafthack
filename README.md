# elysium-crafthack
Agentic AI assisted dashboard containing functional fitness score and core health with sleep, biological age, nutrition etc. 

# Backend Agent API

First, set up virtual environment and install dependencies:
```
# set up virtual environment 
cd ./agent
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.
```

Then run the server with:
```
python main.py
```
We can go to the browser and open http://localhost:9100/docs to see the API documentation and test the endpoints.

# Frontend UI

First, set up virtual environment and install dependencies:
```
cd./ui
npm install
```

Then run the server with:
```
npm run start
```
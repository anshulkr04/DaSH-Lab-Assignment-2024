# Development
## 1. Getting GROQ API KEY
To get started , first get your groq api key from https://console.groq.com/keys and the run the following code:
```
export GROQ_API_KEY={your api key}
```
## 2. Level-1 Asssignment
 
For the Level-1 Assignment
```
cd Level-1
python3 api.py
```
this should overwrite the `output.json` file that already has been created.

## 3. Level-2 Assignment
For the Level-2 Assignment
```
cd ..
cd Level-2
fastapi dev
```
Then open a new terminal
Go to the `DaSH-Lab-Assignment-2024` repo and run the following command
```
source .venv/bin/activate
```
then
```
cd Developement/Level-2
python3 client.py
```
This should create a new `output{clientID}.json` file.
## 4. Script
First provide the necessary permissions using the following command
```
chmod +x script.sh
```
Then run it
```
./script.sh
```
This should create 3 `output{cliendID}.json` files.

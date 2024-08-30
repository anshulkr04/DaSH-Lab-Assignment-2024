#!/bin/bash
cd ..
cd ..
source .venv/bin/activate
cd Development/Level-2
fastapi dev
# Define the directory path and script name
SCRIPT_NAME="client.py"

# Function to open a new terminal, change directory, and run the Python script
run_client() {
    osascript <<EOF
    tell application "Terminal"
        do script "cd DaSH-Lab-Assignment-2024;source .venv/bin/activate;cd Development;cd Level-2; python3 '$SCRIPT_NAME'"
    end tell
EOF
}

# Run the command three times in new terminal windows with a 1-second delay between each
for i in {1..3}
do
    run_client
    sleep 1
done

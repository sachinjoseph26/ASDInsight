#!/bin/sh

# Source the .env file to set environment variables
if [ -f "$ENV_FILE_LOCATION" ]; then
    export $(cat $ENV_FILE_LOCATION | sed 's/#.*//g' | xargs)
fi

# Start the Flask API server
python /app/api_server/manage.py &

# Start the Streamlit application
streamlit run /app/streamlit_ui/streamlit_app.py --server.port 80
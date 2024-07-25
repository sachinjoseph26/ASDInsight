#!/bin/bash

# Start the Flask API server
python /app/api_server/manage.py &

# Start Streamlit app
streamlit run /app/streamlit_ui/streamlit_app.py --server.port 7811
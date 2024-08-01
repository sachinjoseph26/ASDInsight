#!/bin/sh

# Start the Flask API server
python /app/api_server/manage.py &

# Start the Streamlit application
streamlit run /app/streamlit_ui/streamlit_app.py --server.port 7801
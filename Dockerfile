FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code and data into the container
COPY . .

# Hugging Face Spaces requires exposing port 7860
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.port=7860", "--server.address=0.0.0.0"]
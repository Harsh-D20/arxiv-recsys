FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and data into the container
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Run app.py
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
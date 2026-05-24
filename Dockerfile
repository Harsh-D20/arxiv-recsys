# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# We do this first to leverage Docker's caching mechanism
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code and data into the container
COPY app.py .
COPY vector_search.py .
COPY paper_embeddings.npy .
COPY ai_papers_final.parquet .

# 5. Expose the port FastAPI runs on
EXPOSE 8000

# 6. Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
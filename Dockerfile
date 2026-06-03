# Base image Python ringan
FROM python:3.12-slim

# Folder kerja di dalam container
WORKDIR /app

# Install dependency dulu (biar layer-nya ke-cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode project
COPY . .

# Training model saat build -> menghasilkan house_model.pkl di dalam image
# (Kalau Anda sudah punya house_model.pkl dan tidak mau train ulang, hapus baris ini)
RUN python train_model.py

# Port yang dibuka container
EXPOSE 5000

# Jalankan pakai gunicorn (production server), bukan flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

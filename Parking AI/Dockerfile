FROM python:3.10

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/project1/YOLO_Parking1/

COPY requirements1.txt .

RUN pip install -r requirements1.txt

COPY . .

EXPOSE 8183

CMD ["gunicorn", "--bind", "0.0.0.0:8183", "wsgi:application"]

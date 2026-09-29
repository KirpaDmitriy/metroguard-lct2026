FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    METROGUARD_WORK_DIR=/data/jobs

WORKDIR /app
COPY demo_app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY demo_app /app/demo_app
COPY lidar_geometry /app/lidar_geometry
COPY experiments/dashboard.html /app/experiments/dashboard.html

RUN useradd --create-home --uid 10001 app && mkdir -p /data/jobs && chown -R app:app /data
USER app
EXPOSE 8000
CMD ["uvicorn", "demo_app.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]

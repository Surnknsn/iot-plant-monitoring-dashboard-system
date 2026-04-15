# IoT Plant Monitoring Dashboard System

## Overview
This project is an IoT-based plant monitoring system that uses a mecanum robot to capture plant images and analyze them using image processing techniques. The processed data is sent to a real-time dashboard for visualization and monitoring.

## System Architecture
- Mecanum robot controlled by ESP32
- Camera captures plant images
- Image processing using OpenCV
- Data transmitted via HTTP API
- Stored in database (MySQL)
- Visualized in Grafana dashboard

## Features
- Autonomous plant monitoring
- Image thresholding and analysis
- Real-time dashboard visualization
- Remote control via web interface
- Data logging and reporting
- Notification system (Telegram / Email)

## Technologies Used
- ESP32
- OpenCV (Python)
- Node-RED / HTTP API
- MySQL
- Grafana

## System Workflow
1. Robot captures plant image
2. Image is uploaded to server via API
3. Image processing analyzes plant data (area, greenness, etc.)
4. Data is stored in database
5. Dashboard displays real-time data

## Results
- Dashboard showing plant growth data
- Graphs (area, greenness, humidity, seed count)
- Alerts via Telegram and Email

## Project Structure
- /hardware → Robot and ESP32 control
- /image-processing → Image analysis code
- /backend → API and database
- /dashboard → Grafana configuration

## Author
KANASSANAN KONGYOS

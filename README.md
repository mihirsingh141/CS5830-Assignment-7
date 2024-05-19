# CS5830-Assignment-7

## Monitoring the FastAPI Application

This README details the metrics configured in the `main.py` file for posting to Prometheus, enabling extensive monitoring and performance analysis of the FastAPI application.

### Configured Metrics

The following metrics are implemented, each tailored to monitor specific aspects of the application:

- **`api_run_time_milliseconds`**: Measures the API run time in milliseconds.
- **`api_tl_time`**: Tracks the API Time per Length of request in microseconds per character.
- **`api_usage_counter`**: Counts the number of API calls made.
- **`api_memory_usage`**: Monitors the memory usage of the API process.
- **`api_cpu_usage_percent`**: Reports the CPU usage percentage of the API process.
- **`api_network_bytes_sent`**: Logs the network bytes sent by the API process.
- **`api_network_bytes_received`**: Logs the network bytes received by the API process.

### Visualization in Grafana

These metrics are queryable in Grafana, enabling robust visualization and analysis. Users can create dashboards in Grafana to track these metrics in real-time, which is crucial for understanding the performance and resource utilization of the FastAPI application.

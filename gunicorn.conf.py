# Gunicorn configuration file

# Number of worker processes
workers = 4

# The socket to bind
bind = "0.0.0.0:10000"

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Timeout for graceful workers restart
timeout = 30

# Log level
loglevel = 'info'

# Preload the application before forking worker processes
preload_app = True

# Redirect stdout/stderr to log file
capture_output = True

# Use gevent worker class for better performance
worker_class = 'gevent'

# Set the number of threads for handling requests
threads = 4

# Enable hot reload
reload = True
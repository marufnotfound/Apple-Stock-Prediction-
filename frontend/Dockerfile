# Use the official Nginx base image
FROM nginx:alpine

# Remove default Nginx configurations
RUN rm -rf /etc/nginx/conf.d/*

# Copy the custom Nginx configuration file
COPY default.conf /etc/nginx/conf.d/

# Copy the frontend files into the Nginx container
COPY . /usr/share/nginx/html

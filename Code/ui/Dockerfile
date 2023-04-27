# Use Node.js 18.15.0 LTS as the base image
FROM node:18.15.0

# Create and set the working directory for the app
WORKDIR /app

# Copy the package.json and package-lock.json files to the container
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the entire app directory to the container
COPY . .

# Build the app
RUN npm run build

# Expose port 5173 for the app
# EXPOSE 5173

# Start the app
CMD ["npm", "run", "host"]

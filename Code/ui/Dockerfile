# Use a lightweight Node.js image
FROM node:16-alpine

# Set the working directory
WORKDIR /ui

# Copy package.json and package-lock.json files to the container
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code to the container
COPY . .

# Build the application
RUN npm run build

# Start the application
CMD ["npm", "run", "start"]

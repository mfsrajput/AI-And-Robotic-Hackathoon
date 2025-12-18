#!/bin/bash

# Launch Script for Physical AI & Humanoid Robotics Docusaurus Site
# This script checks prerequisites, installs dependencies, and launches the site

set -e  # Exit on any error

# Function to print colored status messages
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Node.js
print_status "Checking for Node.js..."
if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js version 18 or higher."
    print_status "Visit https://nodejs.org/ to download and install Node.js."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | sed 's/v//')
NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)

if [ "$NODE_MAJOR" -lt 18 ]; then
    print_error "Node.js version 18 or higher is required. Current version: $NODE_VERSION"
    exit 1
fi

print_success "Node.js version: $NODE_VERSION (compatible)"

# Check for npm
print_status "Checking for npm..."
if ! command_exists npm; then
    print_error "npm is not installed. Please install npm with Node.js."
    exit 1
fi

print_success "npm is available"

# Check for git
print_status "Checking for git..."
if ! command_exists git; then
    print_error "git is not installed. Please install git."
    exit 1
fi

print_success "git is available"

# Check if we're in the correct directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the root of the project."
    exit 1
fi

print_status "Found package.json in current directory"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_status "node_modules directory not found. Running 'npm install'..."
    npm install
    print_success "Dependencies installed successfully"
else
    print_status "node_modules directory found. Skipping npm install."
fi

# Build the site first to ensure everything is working
print_status "Building the Docusaurus site..."
npm run build
print_success "Site built successfully"

# Start the development server in the background
print_status "Starting the Docusaurus development server..."
npm run start &

# Store the process ID of the server
SERVER_PID=$!

# Wait a moment for the server to start
print_status "Waiting for server to start..."
sleep 5

# Check if the server is running
if ps -p $SERVER_PID > /dev/null; then
    print_success "Docusaurus server is running!"
    print_status "The Physical AI & Humanoid Robotics textbook site is available at:"
    echo ""
    echo -e "  \033[1;36mhttp://localhost:3000\033[0m"
    echo ""
    print_status "Press Ctrl+C to stop the server when you're done."

    # Try to open the browser automatically (works on most systems)
    if command_exists xdg-open; then
        # Linux
        xdg-open http://localhost:3000
    elif command_exists open; then
        # macOS
        open http://localhost:3000
    elif command_exists start; then
        # Windows (Git Bash or WSL)
        start http://localhost:3000
    else
        print_warning "Could not automatically open browser. Please visit http://localhost:3000 manually."
    fi

    # Wait for the server process to finish
    wait $SERVER_PID
else
    print_error "Failed to start the Docusaurus server."
    exit 1
fi
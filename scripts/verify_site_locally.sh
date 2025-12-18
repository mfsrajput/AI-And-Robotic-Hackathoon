#!/bin/bash

# Local Verification Script for Physical AI & Humanoid Robotics Docusaurus Site
# This script verifies that the Docusaurus site is running correctly on http://localhost:3000

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

# Check for required tools
print_status "Checking for required tools..."

if ! command_exists curl; then
    print_error "curl is not installed. Please install curl to run verification checks."
    exit 1
fi

print_success "curl is available"

# Check if launch script exists
if [ ! -f "scripts/launch_textbook_site.sh" ]; then
    print_error "scripts/launch_textbook_site.sh not found."
    exit 1
fi

print_success "Launch script found"

# Start the Docusaurus server in the background
print_status "Starting Docusaurus server..."
bash scripts/launch_textbook_site.sh &
SERVER_PID=$!

# Wait for the server to start
print_status "Waiting for server to start (5 seconds)..."
sleep 5

# Check if the server is running
if ! ps -p $SERVER_PID > /dev/null; then
    print_error "Failed to start the Docusaurus server."
    exit 1
fi

# Function to check if the site is accessible
check_site_accessibility() {
    local max_attempts=30
    local attempt=1

    print_status "Checking if http://localhost:3000 is accessible..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 http://localhost:3000 > /dev/null; then
            print_success "Site is accessible at http://localhost:3000"
            return 0
        else
            print_status "Attempt $attempt/$max_attempts: Site not yet accessible, waiting..."
            sleep 2
        fi
        ((attempt++))
    done

    print_error "Site is not accessible after $max_attempts attempts."
    return 1
}

# Check site accessibility
if check_site_accessibility; then
    print_success "Local verification: Site is running and accessible"

    # Check basic response
    RESPONSE_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
    if [ "$RESPONSE_CODE" -eq 200 ]; then
        print_success "Local verification: Site returns HTTP 200 OK"
    else
        print_error "Local verification: Site returns HTTP $RESPONSE_CODE"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Check for key content on the landing page
    LANDING_CONTENT=$(curl -s http://localhost:3000 | grep -i "Physical AI & Humanoid Robotics" || true)
    if [ -n "$LANDING_CONTENT" ]; then
        print_success "Local verification: Landing page contains expected title"
    else
        print_warning "Local verification: Expected title not found on landing page (may be normal during dev)"
    fi

    # Try to take a screenshot if image capture tools are available
    print_status "Attempting to capture verification screenshot..."

    if command_exists xdg-screenshooter; then
        # Linux with xdg-screenshooter
        xdg-screenshooter -f ./verification_screenshot.png
        print_success "Screenshot saved as verification_screenshot.png"
    elif command_exists gnome-screenshot; then
        # Linux with gnome-screenshot
        gnome-screenshot -f ./verification_screenshot.png
        print_success "Screenshot saved as verification_screenshot.png"
    elif command_exists screencapture; then
        # macOS
        screencapture -w ./verification_screenshot.png
        print_success "Screenshot saved as verification_screenshot.png"
    elif command_exists powershell && command_exists convert; then
        # Windows with PowerShell and ImageMagick
        powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('%{PRTSC}'); Start-Sleep -Seconds 1; Start-Process -FilePath 'magick' -ArgumentList 'clipboard:', './verification_screenshot.png'"
        print_success "Screenshot saved as verification_screenshot.png"
    else
        print_warning "Screenshot tools not available. Please manually capture a screenshot of the site."
        print_status "To manually verify, visit http://localhost:3000 and take a screenshot showing:"
        echo "  - The site title 'Physical AI & Humanoid Robotics'"
        echo "  - The navigation menu with modules"
        echo "  - The 'Start Learning' button on the landing page"
    fi

    # Output next steps
    echo ""
    print_success "==============================================="
    print_success "LOCAL VERIFICATION PASSED"
    print_success "==============================================="
    echo ""
    echo "The Physical AI & Humanoid Robotics textbook site is running correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Complete the manual verification checklist in docs/local-verification-checklist.md"
    echo "2. Run the comprehensive manual tests to ensure all features work"
    echo "3. Document the results in VERIFICATION_REPORT.md"
    echo "4. If all checks pass, the site is ready for release"
    echo ""
    print_status "Manual verification checklist includes:"
    echo "  - Landing page loads with title and 'Start Learning' button"
    echo "  - Sidebar shows all 4 modules with correct chapter titles"
    echo "  - Navigate to at least one chapter in each module"
    echo "  - Mermaid diagrams render correctly"
    echo "  - Code blocks have syntax highlighting"
    echo "  - Responsive on mobile view (resize browser)"
    echo "  - Dark/light mode toggle works"
    echo "  - All links in navbar and sidebar work"
    echo "  - WCAG compliance spot-check (alt-text visible on hover, good contrast)"
    echo ""
    print_status "Press Ctrl+C to stop the server when verification is complete."

    # Keep the server running for manual testing
    print_status "Server is running for manual verification. Keep this terminal open."
    print_status "When finished with manual testing, press Ctrl+C to stop the server."

    # Wait for the server process to finish (user will stop with Ctrl+C)
    wait $SERVER_PID

else
    print_error "Local verification failed: Site is not accessible"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
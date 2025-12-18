# Getting Started with Physical AI & Humanoid Robotics Textbook

Welcome to the "Physical AI & Humanoid Robotics" textbook website! This guide will help you set up and run the educational content on your local machine.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

### Required Software
- **Git** (version 2.0 or higher)
- **Node.js** (version 18.0 or higher)
- **npm** (usually included with Node.js)

### Optional but Recommended
- A modern web browser (Chrome, Firefox, Safari, or Edge)
- A code editor (VS Code, Sublime Text, or similar)

## Installation Steps

### 1. Clone the Repository

Open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/your-organization/AI-And-Robotic-Hackathoon.git
```

Then navigate to the project directory:

```bash
cd AI-And-Robotic-Hackathoon
```

### 2. Verify Prerequisites

Check that you have the required software installed:

```bash
# Check Git version
git --version

# Check Node.js version (should be 18.x or higher)
node --version

# Check npm version
npm --version
```

### 3. Install Dependencies

Install the required packages for the Docusaurus site:

```bash
npm install
```

This command will download and install all necessary dependencies listed in the `package.json` file.

### 4. Launch the Site

You have two options to launch the site:

#### Option A: Using the Launch Script (Recommended)
Run the automated launch script that checks prerequisites and starts the server:

```bash
bash scripts/launch_textbook_site.sh
```

#### Option B: Manual Launch
Run the development server directly:

```bash
npm run start
```

## Accessing the Textbook

Once the server is running, the textbook will be available at:

[http://localhost:3000](http://localhost:3000)

Your browser should open automatically to display the site. If it doesn't, manually navigate to the URL above.

## Site Structure

The textbook is organized into four progressive modules:

1. **Module 1: ROS 2 Fundamentals** - Learn the Robot Operating System
2. **Module 2: Digital Twin** - Explore virtual environments and simulation
3. **Module 3: NVIDIA Isaac** - Leverage AI and simulation platforms
4. **Module 4: Vision-Language-Action System** - Build voice-controlled robots

## Building for Production

To create a production-ready build of the site:

```bash
npm run build
```

This command generates a static site in the `build/` directory that can be deployed to any web server.

## Deployment

To deploy the site to GitHub Pages:

```bash
npm run deploy
```

This command builds the site and pushes it to the `gh-pages` branch for GitHub Pages hosting.

## Troubleshooting

### Common Issues and Solutions

#### Issue: Node.js version is too old
**Error message**: "Node.js version 18 or higher is required"
**Solution**: Update Node.js to version 18 or higher. You can download it from [nodejs.org](https://nodejs.org/)

#### Issue: Port 3000 is already in use
**Error message**: "Port 3000 is not available"
**Solution**: Run the development server on a different port:
```bash
npm run start -- --port 3001
```

#### Issue: npm install fails
**Error message**: Various error messages during installation
**Solution**:
1. Clear npm cache: `npm cache clean --force`
2. Delete node_modules: `rm -rf node_modules package-lock.json`
3. Reinstall: `npm install`

#### Issue: Site doesn't load after starting
**Symptoms**: Browser shows "This site can't be reached" or similar
**Solution**:
1. Check that the development server is running in your terminal
2. Verify the URL is correct (should be http://localhost:3000)
3. Check your firewall settings if applicable

#### Issue: Dependencies are outdated
**Symptoms**: Site loads but has styling issues or broken functionality
**Solution**: Update dependencies:
```bash
npm update
```

### Performance Tips

- If the site loads slowly, try disabling unnecessary browser extensions
- Ensure you have sufficient system memory available
- Close other applications if experiencing performance issues

### Accessibility Features

This textbook site includes several accessibility features:

- **Keyboard Navigation**: All content is accessible via keyboard
- **Screen Reader Support**: Proper semantic markup for screen readers
- **Sufficient Color Contrast**: All text meets WCAG 2.1 AA contrast requirements
- **Alt Text**: All images include descriptive alternative text
- **Responsive Design**: Site works well on various screen sizes

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub repository](https://github.com/your-organization/AI-And-Robotic-Hackathoon) for open issues
2. Review the [Docusaurus documentation](https://docusaurus.io/docs) for additional help
3. Contact the textbook maintainers through the repository

## Next Steps

Now that you have the textbook running locally:

1. Explore the [Introduction](/intro) to understand the course structure
2. Begin with [Module 1: ROS 2 Fundamentals](/docs/01-ros2/intro) if you're new to robotics
3. Follow the learning objectives and exercises in each module
4. Complete the hands-on activities to reinforce your understanding

Happy learning!
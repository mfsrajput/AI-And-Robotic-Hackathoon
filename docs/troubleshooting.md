# Troubleshooting Guide for Physical AI & Humanoid Robotics Textbook Site

This guide addresses common issues that may occur when setting up or running the Docusaurus textbook site.

## Common Build Errors

### "Cannot find module 'prism-react-renderer/themes/github'"

**Problem:** The build fails with an error message indicating that the prism-react-renderer themes cannot be found.

**Cause:** The `prism-react-renderer` package is missing or not properly installed.

**Solution:**
1. Ensure the package is in your `package.json` dependencies:
   ```json
   {
     "dependencies": {
       "prism-react-renderer": "^2.3.0"
     }
   }
   ```

2. If it's missing, add it manually:
   ```bash
   npm install prism-react-renderer
   ```

3. Clear the cache and reinstall:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

4. Verify that your `docusaurus.config.js` imports the themes correctly:
   ```javascript
   module.exports = {
     // ...
     themeConfig: {
       prism: {
         theme: require('prism-react-renderer/themes/github'),
         darkTheme: require('prism-react-renderer/themes/dracula'),
       },
     },
   };
   ```

### "Module not found: Error: Can't resolve '@docusaurus/core'"

**Problem:** The build fails with a module resolution error for Docusaurus core.

**Solution:**
1. Check that `@docusaurus/core` is in your dependencies:
   ```bash
   npm list @docusaurus/core
   ```

2. If missing, install it:
   ```bash
   npm install @docusaurus/core@latest @docusaurus/preset-classic@latest
   ```

### "Port 3000 is not available"

**Problem:** The development server fails to start because port 3000 is already in use.

**Solution:**
1. Find processes using port 3000:
   ```bash
   # Linux/Mac
   lsof -i :3000
   # Windows
   netstat -ano | findstr :3000
   ```

2. Kill the process using port 3000:
   ```bash
   # Linux/Mac (replace PID with actual process ID)
   kill -9 PID
   # Windows (replace PID with actual process ID)
   taskkill /PID PID /F
   ```

3. Or start Docusaurus on a different port:
   ```bash
   npm run start -- --port 3001
   ```

## Performance Issues

### Site Loads Slowly

**Problem:** The site takes a long time to load or seems unresponsive.

**Solution:**
1. Check for large images or assets that may be slowing down the site
2. Ensure you're running in development mode: `npm run start`
3. Clear browser cache and try again
4. Verify that your system meets the minimum requirements

### Long Build Times

**Problem:** The build process takes an excessive amount of time.

**Solution:**
1. Check for circular dependencies in your documentation
2. Ensure you're not importing unnecessary modules
3. Consider using `npm run build` instead of `npm run start` to see actual build times

## Content Issues

### Images Not Loading

**Problem:** Images are not displaying correctly in the documentation.

**Solution:**
1. Verify image paths are correct relative to the markdown file
2. Ensure images are in the `static/img/` directory or in the same directory as the markdown file
3. Check that image file extensions match exactly (case-sensitive)

### Code Blocks Not Highlighted

**Problem:** Code blocks don't have syntax highlighting.

**Solution:**
1. Verify `prism-react-renderer` is installed and in dependencies
2. Check that the language identifier is correct (e.g., ```javascript, ```python)
3. Ensure the prism configuration in `docusaurus.config.js` is correct

## Navigation Issues

### Sidebar Not Showing Correct Items

**Problem:** The sidebar doesn't display all documentation or shows incorrect items.

**Solution:**
1. Verify your `sidebars.js` file has the correct structure
2. Check that `_category_.json` files exist in each documentation directory
3. Ensure all markdown files have proper frontmatter

### Links Not Working

**Problem:** Internal links in the documentation don't navigate correctly.

**Solution:**
1. Verify relative paths are correct
2. Check that linked files exist
3. Ensure URLs use the correct format (e.g., `/docs/intro` not `/intro`)

## Theme and Styling Issues

### Dark/Light Mode Not Working

**Problem:** The theme toggle doesn't switch between dark and light modes.

**Solution:**
1. Verify theme configuration in `docusaurus.config.js`
2. Check that `@docusaurus/theme-classic` is properly configured
3. Ensure no custom CSS is interfering with theme switching

### Custom CSS Not Applied

**Problem:** Custom styles in `src/css/custom.css` are not being applied.

**Solution:**
1. Verify the file is imported in `docusaurus.config.js`:
   ```javascript
   module.exports = {
     presets: [
       [
         'classic',
         {
           theme: {
             customCss: './src/css/custom.css',
           },
         },
       ],
     ],
   };
   ```

## Accessibility Issues

### Missing Alt Text

**Problem:** Images don't have alternative text for screen readers.

**Solution:**
1. Add alt text to all images:
   ```markdown
   ![Description of image](/img/image.png)
   ```

2. For decorative images, use empty alt text:
   ```markdown
   ![](/img/decoration.png)
   ```

### Insufficient Color Contrast

**Problem:** Text doesn't have sufficient contrast against its background.

**Solution:**
1. Use a color contrast checker tool to verify WCAG compliance
2. Adjust colors to meet at least 4.5:1 contrast ratio for normal text
3. For large text, ensure at least 3:1 contrast ratio

## Verification Steps

If you encounter issues, try these verification steps:

1. **Clean Installation:**
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   npm run start
   ```

2. **Check Node Version:**
   ```bash
   node --version  # Should be >= 18.0
   npm --version
   ```

3. **Verify Dependencies:**
   ```bash
   npm ls --depth=0
   ```

4. **Test Build:**
   ```bash
   npm run build
   npm run serve
   ```

## MDX Syntax Issues

### Fixing MDX errors for `<` `>` symbols

**Problem:** MDX v3 compilation errors occur when unescaped `<` and `>` characters appear in plain text (e.g., comparisons like "< 2" being misinterpreted as invalid JSX).

**Cause:** The MDX compiler interprets `<` and `>` as potential JSX tags when they appear in plain text, causing compilation failures.

**Solution:**
1. Wrap comparison expressions in inline code spans:
   ```markdown
   - Instead of: Response time must be < 2 seconds
   - Use: Response time must be `< 2` seconds
   ```

2. Common patterns to fix:
   - `latency < 2s` → `latency &lt; 2s`
   - `error rate <= 5%` → `error rate &lt;= 5%`
   - `threshold > 10` → `threshold &gt; 10`

3. For single symbols that need escaping, you can also use HTML entities:
   - `<` → `&lt;`
   - `>` → `&gt;`

**Important:** Do not modify `<` and `>` characters that are:
- Inside code blocks (``` ... ```)
- Part of JSX/HTML elements
- Within inline code spans (`<code>`)
- Inside Mermaid diagrams

### Displaying literal < > in MDX content

To display literal < > in MDX content, use `&lt;` and `&gt;` or inline code ` < > `

## Landing Page Issues

### Start Learning button goes to 404

**Problem:** The "Start Learning" button on the landing page links to "/docs/intro" which results in a 404 error.

**Cause:** The Docusaurus configuration has routeBasePath set to '/' (docs are at root level), so documentation files are served from the root URL rather than under a "/docs" prefix.

**Solution:**
1. Update the button link in `src/pages/index.md`:
   ```markdown
   - Instead of: href="/docs/intro"
   - Use: href="/intro"
   ```

2. Verify that the intro document exists in the correct location:
   - Should be at `docs/intro.md` (not `docs/docs/intro.md`)

3. Common patterns for similar links:
   - `/docs/intro` → `/intro`
   - `/docs/getting-started` → `/getting-started`
   - `/docs/category/module-1` → `/category/module-1`

## Logo/Favicon Not Showing

**Problem:** The site logo or favicon is not displaying correctly.

**Cause:** Common causes include missing image files, incorrect paths in configuration, or caching issues.

**Solution:**
1. Verify that the logo and favicon files exist in the correct location:
   - Logo: `static/img/logo.svg`
   - Favicon: `static/img/favicon.ico`

2. Check that your `docusaurus.config.js` has the correct paths:
   ```javascript
   const config = {
     // ...
     favicon: 'img/favicon.ico',
     // ...
     themeConfig: {
       // ...
       navbar: {
         // ...
         logo: {
           alt: 'My Site Logo',
           src: 'img/logo.svg',
         },
         // ...
       },
     },
   };
   ```

3. Clear your browser cache and Docusaurus cache:
   ```bash
   # Clear Docusaurus cache
   rm -rf .docusaurus
   # Clear build directory
   rm -rf build
   # Restart development server
   npm run start
   ```

4. Ensure the baseUrl in `docusaurus.config.js` matches your deployment path:
   ```javascript
   const config = {
     // ...
     baseUrl: '/AI-And-Robotic-Hackathoon/',  // Adjust for your deployment
     // ...
   };
   ```

5. Verify the image files are in the correct format and not corrupted.

## Getting Help

If you're still experiencing issues:

1. Check the [Docusaurus documentation](https://docusaurus.io/docs)
2. Look for similar issues in the [GitHub repository](https://github.com/your-organization/AI-And-Robotic-Hackathoon)
3. Verify your Node.js and npm versions meet the requirements
4. Consider opening an issue in the repository if you've found a bug
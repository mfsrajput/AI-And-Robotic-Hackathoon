# Deployment Guide for GitHub Pages

This guide provides step-by-step instructions for deploying the Docusaurus site to GitHub Pages.

## Prerequisites

- Node.js (version >= 18.0)
- npm or yarn package manager
- Git installed and configured
- GitHub repository access with admin permissions

## Deployment Process

### 1. Prepare for Deployment

Before deploying, ensure that:

1. All changes are committed to the main branch
2. The site builds successfully: `npm run build`
3. You have push access to the `gh-pages` branch

### 2. Configure Repository Settings

1. Go to your GitHub repository settings
2. Navigate to "Pages" section
3. Set source to "Deploy from a branch"
4. Select "gh-pages" branch and "/ (root)" folder
5. Save the settings

### 3. Deploy the Site

Run the following command to build and deploy the site:

```bash
npm run deploy
```

This command will:
- Build the static HTML files using `docusaurus build`
- Push the built files to the `gh-pages` branch
- GitHub Pages will automatically serve the site

### 4. Verify Deployment

After deployment, your site will be available at: https://mfsrajput.github.io/AI-And-Robotic-Hackathoon/

## Configuration Details

The deployment is configured in:

- `docusaurus.config.js`: Contains `organizationName` and `projectName` for GitHub Pages
- `package.json`: Contains the deploy script that uses `docusaurus deploy`

### Key Configuration Values

- **organizationName**: `mfsrajput`
- **projectName**: `AI-And-Robotic-Hackathoon`
- **baseUrl**: `/AI-And-Robotic-Hackathoon/`
- **URL**: `https://mfsrajput.github.io`

## Troubleshooting

### Site Not Updating

If the site doesn't update after deployment:

1. Clear Docusaurus cache: `npm run clear`
2. Rebuild the site: `npm run build`
3. Deploy again: `npm run deploy`

### GitHub Pages Not Activated

If GitHub Pages shows as disabled:

1. Check repository settings to ensure GitHub Pages is enabled
2. Verify the source branch is set to `gh-pages`
3. Confirm the repository is public (required for GitHub Pages)

### Custom Domain

If you want to use a custom domain:

1. Add a `CNAME` file in the `static/` directory with your domain
2. Update the `url` field in `docusaurus.config.js`
3. Configure DNS settings with your domain provider

## Best Practices

- Always test locally before deploying: `npm run start`
- Build and test the production version: `npm run build && npm run serve`
- Keep dependencies up to date
- Verify all links work properly after deployment
- Monitor the deployment process for any errors

## Security Considerations

- Ensure sensitive information is not included in the build
- Regularly update dependencies to patch security vulnerabilities
- Review all changes before committing to the main branch
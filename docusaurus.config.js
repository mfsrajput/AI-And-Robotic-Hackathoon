// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to understand your code
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive textbook on building intelligent humanoid robots',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://mfsrajput.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<organization-name>/'
  // TEMPORARY: Set to '/' for local dev, switch back to '/AI-And-Robotic-Hackathoon/' for production
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'mfsrajput', // Usually your GitHub org/user name.
  projectName: 'AI-And-Robotic-Hackathoon', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: '/', // Serve docs at root
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/mfsrajput/AI-And-Robotic-Hackathoon/edit/main/',
          // Add remark and rehype plugins to handle potential MDX issues
          remarkPlugins: [
            // Additional remark plugins can be added here if needed
          ],
          rehypePlugins: [
            // Additional rehype plugins can be added here if needed
          ],
        },
        blog: false, // Disable blog functionality
        theme: {
          customCss: [
            './src/css/custom.css',
            './src/css/chatbot.css' // Include chatbot CSS
          ],
        },
      }),
    ],
  ],

  themes: [
    // Add the @docusaurus/theme-live-codeblock theme if you want live codeblocks
  ],

  plugins: [
    // Remove the duplicate docs plugin that may be causing conflicts
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            type: 'docSidebar',
            sidebarId: 'moduleSidebar',
            label: 'Modules',
            position: 'left'
          },
          {
            href: 'https://github.com/mfsrajput/AI-And-Robotic-Hackathoon',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook',
            items: [
              {
                label: 'Introduction',
                to: '/intro',
              },
              {
                label: 'Module 1: ROS 2 Fundamentals',
                to: '/category/module-1-the-robotic-nervous-system-ros-2',
              },
              {
                label: 'Module 2: Digital Twin',
                to: '/category/module-2-the-digital-twin-gazebo--unity',
              },
              {
                label: 'Module 3: NVIDIA Isaac',
                to: '/category/module-3-the-ai-robot-brain-nvidia-isaac',
              },
              {
                label: 'Module 4: VLA System',
                to: '/category/module-4-vision-language-action-vla--capstone',
              },
              {
                label: 'AI Assistant',
                to: '/chatbot',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/mfsrajput/AI-And-Robotic-Hackathoon',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Preface',
                to: '/preface',
              },
              {
                label: 'Contributors',
                to: '/contributors',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer/themes/github'),
        darkTheme: require('prism-react-renderer/themes/dracula'),
      },
    }),

  // Add custom scripts to inject the chatbot into all pages
  // scripts: [
  //   {
  //     src: '/js/chatbot-injector.js', // This will be a custom script to handle the integration
  //     async: true,
  //     defer: true,
  //   },
  // ],

  // Add polyfills for local development
  scripts: [
    {
      src: '/src/polyfills.js',
      async: false, // Load synchronously to ensure process is available early
    },
  ],
};

module.exports = config;
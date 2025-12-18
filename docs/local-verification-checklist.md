# Local Verification Checklist for Physical AI & Humanoid Robotics Textbook Site

This checklist ensures that the Docusaurus textbook site is fully functional before public release. Complete each item to verify the site works correctly on http://localhost:3000.

## Pre-Verification Setup

- [ ] Start the local server using: `bash scripts/launch_textbook_site.sh`
- [ ] Wait for the message "Docusaurus server is running!" before proceeding
- [ ] Open your browser to http://localhost:3000

## Landing Page Verification

- [ ] Landing page loads correctly with title "Physical AI & Humanoid Robotics"
- [ ] Subtitle "Building Intelligent Humanoid Robots for the Future" is visible
- [ ] "Start Learning" button is visible and clickable
- [ ] Button takes you to the introduction page when clicked
- [ ] All introductory text is readable and properly formatted
- [ ] Navigation menu is visible at the top of the page

## Navigation and Sidebar Verification

- [ ] Sidebar shows all 4 modules: "Module 1: ROS 2 Fundamentals", "Module 2: Digital Twin", "Module 3: NVIDIA Isaac Sim", "Module 4: Vision-Language-Action (VLA) System"
- [ ] Each module expands to show correct chapter titles when clicked
- [ ] All 4 modules are listed in the sidebar
- [ ] Navigation menu at the top works (Textbook, Modules, GitHub links)
- [ ] Footer links work correctly (Introduction, Module links, Contributors, etc.)

## Module Content Verification

- [ ] Navigate to at least one chapter in Module 1 (ROS 2 Fundamentals)
- [ ] Navigate to at least one chapter in Module 2 (Digital Twin)
- [ ] Navigate to at least one chapter in Module 3 (NVIDIA Isaac)
- [ ] Navigate to at least one chapter in Module 4 (VLA System)
- [ ] All chapter pages load without errors
- [ ] Content displays correctly in each module
- [ ] Learning objectives are properly formatted if present
- [ ] Code examples are visible and readable

## Visual Elements Verification

- [ ] Mermaid diagrams render correctly (if present in any chapter)
- [ ] Code blocks have proper syntax highlighting
- [ ] Code block copy buttons work (if present)
- [ ] Images load correctly and are not broken
- [ ] Mathematical equations (if any) render properly
- [ ] Tables display correctly with proper formatting
- [ ] All diagrams and visual elements are visible and clear

## Responsiveness and UI Features

- [ ] Site is responsive on mobile view (resize browser to `&lt;` 768px width)
- [ ] Navigation menu converts to hamburger menu on small screens
- [ ] Text remains readable on mobile view
- [ ] All interactive elements are accessible on mobile view
- [ ] Dark/light mode toggle works (top-right corner)
- [ ] Theme switcher properly changes the site appearance
- [ ] All content remains readable in both themes

## Link Verification

- [ ] All links in the navbar work correctly
- [ ] All links in the sidebar work correctly
- [ ] Internal links within chapters work (if any)
- [ ] External links (GitHub, etc.) open in new tabs (if configured)
- [ ] "Start Learning" button on landing page works
- [ ] Footer links all work correctly

## Accessibility Verification (WCAG Compliance)

- [ ] Alt-text is visible when hovering over images (or inspect element to check)
- [ ] Text has good contrast against background (especially important in dark mode)
- [ ] All interactive elements have visible focus states
- [ ] Site is navigable using keyboard only (try Tab navigation)
- [ ] Headings follow proper hierarchy (H1, H2, H3, etc.)
- [ ] Links have descriptive text (not just "click here")
- [ ] Form elements (if any) have proper labels

## Performance and Loading

- [ ] Pages load quickly (under 3 seconds)
- [ ] No console errors appear when loading pages (check browser dev tools)
- [ ] All assets (images, CSS, JS) load correctly
- [ ] Site doesn't have any broken links or missing resources
- [ ] Code blocks don't cause horizontal scrolling issues

## Content Verification

- [ ] Introduction page content is complete and properly formatted
- [ ] Preface page is accessible and readable
- [ ] Contributors page displays correctly
- [ ] All 4 modules have content and are not empty
- [ ] Chapter content matches the expected educational material
- [ ] All learning objectives are clearly stated
- [ ] Exercises and examples are properly formatted

## Browser Compatibility

- [ ] Site works in Chrome/Chromium
- [ ] Site works in Firefox
- [ ] Site works in Safari (if available)
- [ ] Site works in Edge (if available)
- [ ] No browser-specific rendering issues

## Final Verification

- [ ] Site URL shows http://localhost:3000
- [ ] All content is readable and properly formatted
- [ ] No broken elements or missing resources
- [ ] All interactive features work as expected
- [ ] Mobile responsiveness verified
- [ ] Accessibility features working

## Post-Verification

- [ ] Document any issues found during verification
- [ ] Report bugs to the development team if any are found
- [ ] Complete the VERIFICATION_REPORT.md with results
- [ ] Confirm the site is ready for public release if all checks pass

---

**Verification completed by**: [Your name]

**Date**: [Date]

**Browser used**: [Browser name and version]

**Operating System**: [OS and version]

**Notes**: [Any additional observations or issues]
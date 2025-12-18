"""
Accessibility compliance utilities for WCAG 2.1 AA verification
Tools to ensure educational content meets accessibility standards
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import html
from bs4 import BeautifulSoup
import markdown
from color_utils import ColorContrastChecker  # We'll create this utility


@dataclass
class AccessibilityIssue:
    """Structure for accessibility issues"""
    id: str
    severity: str  # 'critical', 'error', 'warning', 'notice'
    wcag_level: str  # 'A', 'AA', 'AAA'
    guidelines: List[str]  # Related WCAG guidelines
    element: str  # The problematic element
    description: str
    suggestion: str
    context: str


class ColorContrastChecker:
    """
    Check color contrast ratios according to WCAG 2.1 standards
    """
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB values"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_luminance(r: int, g: int, b: int) -> float:
        """Calculate relative luminance according to WCAG formula"""
        def srgb_to_linear(color_val):
            color_val /= 255.0
            return color_val / 12.92 if color_val <= 0.04045 else ((color_val + 0.055) / 1.055) ** 2.4
        
        r_lin = srgb_to_linear(r)
        g_lin = srgb_to_linear(g)
        b_lin = srgb_to_linear(b)
        
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    
    @classmethod
    def calculate_contrast_ratio(cls, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors"""
        r1, g1, b1 = cls.hex_to_rgb(color1)
        r2, g2, b2 = cls.hex_to_rgb(color2)
        
        lum1 = cls.rgb_to_luminance(r1, g1, b1)
        lum2 = cls.rgb_to_luminance(r2, g2, b2)
        
        # Ensure lum1 is the lighter color
        if lum2 > lum1:
            lum1, lum2 = lum2, lum1
        
        return (lum1 + 0.05) / (lum2 + 0.05)
    
    @classmethod
    def check_contrast(cls, fg_color: str, bg_color: str, text_size: str = 'normal') -> Tuple[bool, str]:
        """Check if color contrast meets WCAG 2.1 AA requirements"""
        contrast_ratio = cls.calculate_contrast_ratio(fg_color, bg_color)
        
        if text_size == 'large' or text_size == 'bold_large':  # 18pt+ or 14pt+ bold
            # WCAG AA requires 3:1 for large text
            meets_aa = contrast_ratio >= 3.0
            required_ratio = "3:1"
        else:  # Normal text
            # WCAG AA requires 4.5:1 for normal text
            meets_aa = contrast_ratio >= 4.5
            required_ratio = "4.5:1"
        
        status = "PASS" if meets_aa else "FAIL"
        return meets_aa, f"{status} (ratio: {contrast_ratio:.2f}, required: {required_ratio})"


class AccessibilityChecker:
    """
    Comprehensive accessibility checker for educational content
    """
    
    def __init__(self):
        self.issues: List[AccessibilityIssue] = []
        self.color_checker = ColorContrastChecker()
    
    def check_markdown_accessibility(self, markdown_content: str, context: str = "") -> List[AccessibilityIssue]:
        """Check accessibility of markdown content"""
        self.issues = []  # Reset issues
        
        # Convert markdown to HTML for analysis
        html_content = markdown.markdown(markdown_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check for missing alt text on images
        self._check_missing_alt_text(soup, context)
        
        # Check for heading structure
        self._check_heading_structure(soup, context)
        
        # Check for color contrast (if color information is available)
        self._check_color_contrast(soup, context)
        
        # Check for proper link text
        self._check_link_text(soup, context)
        
        # Check for form elements (if any)
        self._check_form_elements(soup, context)
        
        return self.issues
    
    def _check_missing_alt_text(self, soup: BeautifulSoup, context: str):
        """Check for images without alt text"""
        images = soup.find_all('img')
        for img in images:
            alt_text = img.get('alt', '').strip()
            if not alt_text:
                issue = AccessibilityIssue(
                    id=f"img-alt-missing-{len(self.issues)+1}",
                    severity='error',
                    wcag_level='A',
                    guidelines=['1.1.1'],
                    element=str(img),
                    description="Image missing alt text",
                    suggestion="Add descriptive alt text to the image",
                    context=context
                )
                self.issues.append(issue)
            elif len(alt_text) < 5:
                issue = AccessibilityIssue(
                    id=f"img-alt-too-short-{len(self.issues)+1}",
                    severity='warning',
                    wcag_level='A',
                    guidelines=['1.1.1'],
                    element=str(img),
                    description="Image alt text is too short to be descriptive",
                    suggestion="Make alt text more descriptive of the image content",
                    context=context
                )
                self.issues.append(issue)
    
    def _check_heading_structure(self, soup: BeautifulSoup, context: str):
        """Check for proper heading hierarchy"""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        expected_level = 1
        previous_level = 0
        
        for heading in headings:
            current_level = int(heading.name[1])  # Extract number from h1, h2, etc.
            
            # Check if heading levels skip (e.g., h1 followed by h3)
            if current_level > previous_level + 1 and previous_level != 0:
                issue = AccessibilityIssue(
                    id=f"heading-skip-{len(self.issues)+1}",
                    severity='warning',
                    wcag_level='AA',
                    guidelines=['1.3.1'],
                    element=str(heading),
                    description=f"Heading level skipped from h{previous_level} to h{current_level}",
                    suggestion=f"Use proper heading hierarchy (h{previous_level} should be followed by h{previous_level+1})",
                    context=context
                )
                self.issues.append(issue)
            
            previous_level = current_level
    
    def _check_color_contrast(self, soup: BeautifulSoup, context: str):
        """Check color contrast of text elements"""
        # This is a simplified check - in practice, you'd need to parse CSS
        text_elements = soup.find_all(['p', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'a'])
        
        for elem in text_elements:
            style = elem.get('style', '')
            if 'color' in style and 'background' in style:
                # Extract color values (simplified - real implementation would be more robust)
                color_match = re.search(r'color:\s*([^;]+)', style)
                bg_match = re.search(r'background(-color)?:\s*([^;]+)', style)
                
                if color_match and bg_match:
                    fg_color = color_match.group(1).strip()
                    bg_color = bg_match.group(2).strip()
                    
                    # Convert to hex if needed (simplified)
                    if fg_color.startswith('rgb'):
                        # Convert RGB to hex (simplified conversion)
                        continue
                    
                    if fg_color.startswith('#') and bg_color.startswith('#'):
                        is_valid, result = self.color_checker.check_contrast(fg_color, bg_color)
                        if not is_valid:
                            issue = AccessibilityIssue(
                                id=f"color-contrast-{len(self.issues)+1}",
                                severity='error',
                                wcag_level='AA',
                                guidelines=['1.4.3'],
                                element=str(elem),
                                description=f"Insufficient color contrast: {result}",
                                suggestion="Adjust foreground or background colors to meet contrast requirements",
                                context=context
                            )
                            self.issues.append(issue)
    
    def _check_link_text(self, soup: BeautifulSoup, context: str):
        """Check for meaningful link text"""
        links = soup.find_all('a', href=True)
        
        for link in links:
            link_text = link.get_text().strip()
            
            # Check for non-descriptive link text
            non_descriptive = ['click here', 'here', 'more', 'read more', 'link', 'this']
            if any(text in link_text.lower() for text in non_descriptive):
                issue = AccessibilityIssue(
                    id=f"link-text-descriptive-{len(self.issues)+1}",
                    severity='warning',
                    wcag_level='A',
                    guidelines=['2.4.4'],
                    element=str(link),
                    description=f"Non-descriptive link text: '{link_text}'",
                    suggestion="Use descriptive link text that indicates where the link goes",
                    context=context
                )
                self.issues.append(issue)
            
            # Check for empty link text
            if not link_text:
                issue = AccessibilityIssue(
                    id=f"link-text-empty-{len(self.issues)+1}",
                    severity='error',
                    wcag_level='A',
                    guidelines=['2.4.4'],
                    element=str(link),
                    description="Link with no text content",
                    suggestion="Add descriptive text to the link",
                    context=context
                )
                self.issues.append(issue)
    
    def _check_form_elements(self, soup: BeautifulSoup, context: str):
        """Check form elements for accessibility"""
        inputs = soup.find_all(['input', 'textarea', 'select'])
        
        for input_elem in inputs:
            # Check for associated labels
            input_id = input_elem.get('id')
            if input_id:
                label = soup.find('label', attrs={'for': input_id})
                if not label:
                    issue = AccessibilityIssue(
                        id=f"form-label-missing-{len(self.issues)+1}",
                        severity='error',
                        wcag_level='A',
                        guidelines=['1.3.1', '3.3.2'],
                        element=str(input_elem),
                        description=f"Input field '{input_id}' missing associated label",
                        suggestion="Add a label element associated with the input using 'for' attribute",
                        context=context
                    )
                    self.issues.append(issue)
    
    def check_document_structure(self, content: str, context: str = "") -> List[AccessibilityIssue]:
        """Check document structure accessibility"""
        issues = []
        
        # Check for document language
        if '<html' not in content.lower() or 'lang=' not in content.lower():
            issue = AccessibilityIssue(
                id=f"doc-lang-missing-{len(issues)+1}",
                severity='warning',
                wcag_level='A',
                guidelines=['3.1.1'],
                element="document",
                description="Document language not specified",
                suggestion="Add lang attribute to html element (e.g., <html lang='en'>)",
                context=context
            )
            issues.append(issue)
        
        # Check for skip navigation link
        if 'skip' not in content.lower():
            issue = AccessibilityIssue(
                id=f"skip-nav-missing-{len(issues)+1}",
                severity='warning',
                wcag_level='A',
                guidelines=['2.4.1'],
                element="document",
                description="Skip navigation link not provided",
                suggestion="Add a 'skip to content' link for keyboard users",
                context=context
            )
            issues.append(issue)
        
        self.issues.extend(issues)
        return issues
    
    def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive accessibility report"""
        severity_counts = {'critical': 0, 'error': 0, 'warning': 0, 'notice': 0}
        level_counts = {'A': 0, 'AA': 0, 'AAA': 0}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            level_counts[issue.wcag_level] += 1
        
        compliance_status = "PASS" if severity_counts['error'] == 0 and severity_counts['critical'] == 0 else "FAIL"
        
        return {
            'total_issues': len(self.issues),
            'severity_counts': severity_counts,
            'level_counts': level_counts,
            'compliance_status': compliance_status,
            'issues': [vars(issue) for issue in self.issues],
            'summary': self._generate_summary(severity_counts, level_counts)
        }
    
    def _generate_summary(self, severity_counts: Dict[str, int], level_counts: Dict[str, int]) -> str:
        """Generate a summary of accessibility findings"""
        total_issues = sum(severity_counts.values())
        
        if total_issues == 0:
            return "No accessibility issues found. Content appears to meet WCAG 2.1 AA standards."
        
        summary = f"Found {total_issues} accessibility issues:\n"
        summary += f"  Critical: {severity_counts['critical']}\n"
        summary += f"  Errors: {severity_counts['error']}\n"
        summary += f"  Warnings: {severity_counts['warning']}\n"
        summary += f"  Notices: {severity_counts['notice']}\n"
        
        if severity_counts['error'] > 0 or severity_counts['critical'] > 0:
            summary += "\n⚠️  Content does not meet WCAG 2.1 AA compliance. Issues need to be addressed."
        else:
            summary += "\n⚠️  Content has some accessibility warnings that should be addressed for full compliance."
        
        return summary


class WCAG21AAValidator:
    """
    Validator specifically for WCAG 2.1 AA compliance
    """
    
    def __init__(self):
        self.checker = AccessibilityChecker()
    
    def validate_educational_content(self, content: str, content_type: str = "markdown", context: str = "") -> Dict[str, Any]:
        """Validate educational content for WCAG 2.1 AA compliance"""
        
        if content_type.lower() == "markdown":
            issues = self.checker.check_markdown_accessibility(content, context)
        else:
            issues = self.checker.check_document_structure(content, context)
        
        report = self.checker.generate_accessibility_report()
        report['is_compliant'] = report['compliance_status'] == 'PASS'
        
        return report
    
    def check_specific_guidelines(self) -> Dict[str, str]:
        """Check compliance with specific WCAG 2.1 AA guidelines"""
        guidelines_status = {
            # Perceivable
            '1.1.1': 'Non-text Content - Provide text alternatives',
            '1.2.2': 'Captions (Prerecorded) - Provide captions for audio content',
            '1.3.1': 'Info and Relationships - Use proper markup structure',
            '1.4.3': 'Contrast (Minimum) - Maintain sufficient color contrast',
            '1.4.12': 'Text Spacing - Allow text spacing adjustment',
            
            # Operable
            '2.1.1': 'Keyboard - Ensure full keyboard operability',
            '2.4.1': 'Bypass Blocks - Provide ways to skip repeated content',
            '2.4.4': 'Link Purpose - Make link purpose clear from context',
            '2.5.3': 'Label in Name - Ensure accessible names match visible labels',
            
            # Understandable
            '3.1.1': 'Language of Page - Specify document language',
            '3.2.2': 'On Input - Changes of context are initiated by user action',
            '3.3.2': 'Labels or Instructions - Provide labels for form controls',
            
            # Robust
            '4.1.2': 'Name, Role, Value - Ensure elements have complete information'
        }
        
        return guidelines_status
    
    def get_implementation_tips(self) -> List[str]:
        """Get tips for implementing WCAG 2.1 AA compliance"""
        return [
            "Use proper heading hierarchy (h1, h2, h3, etc.)",
            "Provide alternative text for all images",
            "Ensure sufficient color contrast (4.5:1 for normal text, 3:1 for large text)",
            "Use descriptive link text instead of 'click here'",
            "Provide captions for audio and video content",
            "Make sure all functionality is available from keyboard",
            "Use ARIA labels when necessary for complex widgets",
            "Provide skip navigation links for keyboard users",
            "Ensure forms have proper labels and error handling",
            "Use semantic HTML elements appropriately"
        ]


def create_educational_alt_text(image_description: str, content_purpose: str) -> str:
    """
    Helper function to create educational alt text
    """
    if content_purpose == "diagram":
        return f"Diagram showing {image_description}. This illustrates key concepts for educational purposes."
    elif content_purpose == "code_example":
        return f"Code example demonstrating {image_description}. Shows proper implementation for students."
    elif content_purpose == "process_flow":
        return f"Process flow diagram for {image_description}. Shows the sequence of operations step by step."
    else:
        return f"Educational content: {image_description}. Provides visual learning support for the topic."


def main():
    """Main function to demonstrate accessibility checking"""
    print("WCAG 2.1 AA Accessibility Checker for Educational Content")
    print("="*65)
    
    # Example markdown content to test
    test_content = """
# Introduction to Voice Commands

![Voice Processing Diagram](voice-diagram.png)

This module covers voice-to-text processing using Whisper.

## Steps:
1. Audio input
2. Preprocessing  
3. Model inference
4. Text output

For more information, [click here](#info).
"""
    
    validator = WCAG21AAValidator()
    report = validator.validate_educational_content(test_content, "markdown", "test_module")
    
    print(report['summary'])
    print(f"\nCompliance Status: {'✅ PASS' if report['is_compliant'] else '❌ FAIL'}")
    
    print("\nWCAG 2.1 AA Implementation Tips:")
    for i, tip in enumerate(validator.get_implementation_tips(), 1):
        print(f"  {i}. {tip}")
    
    print(f"\nTotal Issues Found: {report['total_issues']}")
    if report['issues']:
        print("\nSample Issues:")
        for issue in report['issues'][:3]:  # Show first 3 issues
            print(f"  - {issue['description']}")


if __name__ == "__main__":
    main()

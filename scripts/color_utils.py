"""
Color utilities for accessibility checking
Provides color contrast analysis for WCAG 2.1 compliance
"""

from typing import Tuple


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


# Example usage and testing
if __name__ == "__main__":
    checker = ColorContrastChecker()
    
    # Test some color combinations
    test_colors = [
        ("#000000", "#ffffff"),  # Black on white
        ("#ffffff", "#000000"),  # White on black
        ("#0000ff", "#ffffcc"),  # Blue on light yellow
        ("#808080", "#ffffff"),  # Gray on white (should pass for large text, fail for normal)
    ]
    
    print("Color Contrast Checker - WCAG 2.1 AA Compliance")
    print("="*50)
    
    for fg, bg in test_colors:
        # Check for normal text
        meets_normal, result_normal = checker.check_contrast(fg, bg, 'normal')
        # Check for large text
        meets_large, result_large = checker.check_contrast(fg, bg, 'large')
        
        print(f"{fg} on {bg}:")
        print(f"  Normal text: {result_normal}")
        print(f"  Large text:  {result_large}")
        print()

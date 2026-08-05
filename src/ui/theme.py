"""Design System and Theme configuration."""
from __future__ import annotations

import flet as ft
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Palette:
    background: str = "#050505"  # Pure deep dark
    surface: str = "#0D0D12"     # Dark slate
    surface_alt: str = "#16161D"  # Elevated slate
    panel: str = "#0A0A0F"
    panel_alt: str = "#12121A"
    ink: str = "#F8FAFC"         # Crisp white
    muted: str = "#94A3B8"       # Subtle gray
    line: str = "#27272A"        # Zinc 800 for subtle borders
    primary: str = "#8B5CF6"     # Vibrant Violet
    primary_deep: str = "#7C3AED" # Deep Violet
    primary_soft: str = "#4C1D95" # Dark Violet
    accent: str = "#06B6D4"      # Cyan / Neon Blue
    accent_soft: str = "#164E63"  # Deep Cyan
    success: str = "#10B981"     # Emerald
    warning: str = "#F59E0B"     # Amber
    danger: str = "#F43F5E"      # Rose (premium red)
    hero_start: str = "#2E1065"  # Violet 950
    hero_end: str = "#020617"    # Slate 950
    ai_accent: str = "#D946EF"   # Fuchsia for DeepSeek highlights

PALETTE = Palette()

def get_theme() -> ft.Theme:
    """Return the modern Flet theme configuration."""
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            background=PALETTE.background,
            surface=PALETTE.surface,
            primary=PALETTE.primary,
            on_primary=PALETTE.ink,
            secondary=PALETTE.accent,
            error=PALETTE.danger,
        ),
        font_family="Inter",
        use_material3=True,
    )

def setup_page(page: ft.Page) -> None:
    """Apply global theme and settings to the Flet page."""
    page.theme = get_theme()
    page.bgcolor = PALETTE.background
    page.padding = 0
    # Enable Google Fonts
    page.fonts = {
        "Inter": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        "Outfit": "https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap"
    }

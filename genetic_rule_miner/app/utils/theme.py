from flet import Colors, ColorScheme, Page, Theme, ThemeMode


def setup_theme(page: Page):
    page.bgcolor = "#282A36" if page.theme_mode == "dark" else Colors.WHITE

    """
    Configures the theme for the Flet app using the light theme by default.
    """
    page.theme_mode = ThemeMode.LIGHT
    page.theme = get_light_theme()
    page.dark_theme = get_dark_theme()
    page.update()


def get_light_theme() -> Theme:
    """
    Returns a custom light theme.
    """
    return Theme(
        color_scheme=ColorScheme(  # type: ignore
            background="#FFFFFF",
            on_background=Colors.BLACK,
            surface="#FFFFFF",
            on_surface=Colors.BLACK,
            primary=Colors.PURPLE_400,
            secondary=Colors.PURPLE_400,
            surface_variant="#E0E0E0",
            on_surface_variant=Colors.BLACK,
            error=Colors.RED,
            on_error=Colors.WHITE,
            error_container=Colors.RED_200,
            on_error_container=Colors.WHITE,
        )
    )


def get_dark_theme() -> Theme:
    """
    Returns a custom dark theme with improved contrast.
    """
    return Theme(
        color_scheme=ColorScheme(  # type: ignore
            background="#282A36",
            on_background="#FFFFFF",
            surface="#44475A",
            on_surface="#F0F0F0",
            primary=Colors.PURPLE_400,
            secondary="#B39DDB",
            surface_variant="#323450",
            on_surface_variant="#F0F0F0",
            error=Colors.RED,
            on_error="#FFFFFF",
            error_container=Colors.RED_200,
            on_error_container="#FFFFFF",
            outline_variant="#282A36",
        )
    )

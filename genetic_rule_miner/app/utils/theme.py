from flet import Colors, ColorScheme, Page, Theme


def setup_theme(page: Page):
    page.bgcolor = "#282A36" if page.theme_mode == "dark" else Colors.WHITE

    """
    Configures the theme for the Flet app using the light theme by default.
    """
    page.theme_mode = "light"
    page.theme = get_light_theme()
    page.dark_theme = get_dark_theme()
    page.update()


def get_light_theme():
    """
    Returns a custom light theme.
    """
    return Theme(
        color_scheme=ColorScheme(
            background="#FFFFFF",
            on_background=Colors.BLACK,
            surface="#FFFFFF",
            on_surface=Colors.BLACK,
            primary=Colors.PURPLE,
            secondary=Colors.CYAN,
            surface_variant="#E0E0E0",
            on_surface_variant=Colors.BLACK,
            error=Colors.RED,
            on_error=Colors.WHITE,
            error_container=Colors.RED_200,
            on_error_container=Colors.WHITE,
        )
    )


def get_dark_theme():
    """
    Returns a custom dark theme.
    """
    return Theme(
        color_scheme=ColorScheme(
            background="#FFFFFF",
            on_background=Colors.WHITE,
            surface="#282A36",
            on_surface=Colors.WHITE,
            primary=Colors.PURPLE,
            secondary=Colors.CYAN,
            surface_variant="#383A59",
            on_surface_variant=Colors.WHITE,
            error=Colors.RED,
            on_error=Colors.WHITE,
            error_container=Colors.RED_200,
            on_error_container=Colors.WHITE,
        ),
    )

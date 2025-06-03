import flet as ft


def top_app_bar(page):
    def toggle_theme(e):
        page.theme_mode = (
            ft.ThemeMode.LIGHT
            if page.theme_mode == ft.ThemeMode.DARK
            else ft.ThemeMode.DARK
        )
        page.update()

    def minimize_window(e):
        page.window.minimized = True
        page.update()

    def toggle_maximize(e):
        page.window.maximized = not page.window.maximized
        page.update()

    def close_window(e):
        page.window.close()

    # Envolver el contenido en WindowDragArea
    return ft.WindowDragArea(
        content=ft.Row(
            [
                # Lado izquierdo - botón de tema
                ft.IconButton(
                    icon=ft.Icons.BRIGHTNESS_6_ROUNDED,
                    tooltip="Toggle theme",
                    on_click=toggle_theme,
                ),
                # Centro - título
                ft.Text(
                    "Series Recommender", size=20, weight=ft.FontWeight.BOLD
                ),
                # Lado derecho - controles de ventana
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.MINIMIZE,
                            tooltip="Minimize",
                            on_click=minimize_window,
                            icon_size=18,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CROP_SQUARE,
                            tooltip="Maximize/Restore",
                            on_click=toggle_maximize,
                            icon_size=18,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            tooltip="Close",
                            on_click=close_window,
                            icon_size=18,
                        ),
                    ],
                    spacing=0,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
    )

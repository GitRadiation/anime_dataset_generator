import flet as ft


def top_app_bar(page):
    def toggle_theme(e):
        page.theme_mode = "light" if page.theme_mode == "dark" else "dark"
        page.update()

    return ft.Row(
        [
            ft.IconButton(
                icon=ft.Icons.BRIGHTNESS_6_ROUNDED,
                tooltip="Toggle theme",
                on_click=toggle_theme,
            ),
            ft.Text("Series Recommender", size=20, weight=ft.FontWeight.BOLD),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )

import flet as ft


def top_app_bar(page, translations):
    lang = page.session.get("lang") or "en"

    t = translations.get(lang, translations["en"])

    def toggle_theme(e):
        page.theme_mode = (
            ft.ThemeMode.LIGHT
            if page.theme_mode == ft.ThemeMode.DARK
            else ft.ThemeMode.DARK
        )
        page.bgcolor = (
            "#282A36"
            if page.theme_mode == ft.ThemeMode.DARK
            else ft.Colors.WHITE
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

    window_controls = []
    if not page.web:
        window_controls = [
            ft.IconButton(
                icon=ft.Icons.MINIMIZE,
                tooltip=t["minimize"],
                on_click=minimize_window,
                icon_size=16,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=4),
                    overlay_color={
                        ft.ControlState.HOVERED: ft.Colors.with_opacity(
                            0.1, ft.Colors.ON_SURFACE
                        ),
                    },
                ),
            ),
            ft.IconButton(
                icon=ft.Icons.CROP_SQUARE,
                tooltip=t["maximize"],
                on_click=toggle_maximize,
                icon_size=16,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=4),
                    overlay_color={
                        ft.ControlState.HOVERED: ft.Colors.with_opacity(
                            0.1, ft.Colors.ON_SURFACE
                        ),
                    },
                ),
            ),
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip=t["close"],
                on_click=close_window,
                icon_size=16,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=4),
                    overlay_color={
                        ft.ControlState.HOVERED: ft.Colors.with_opacity(
                            0.8, ft.Colors.RED_400
                        ),
                    },
                ),
            ),
        ]

    row_content = ft.Row(
        [
            ft.Container(
                content=ft.IconButton(
                    icon=ft.Icons.BRIGHTNESS_6_ROUNDED,
                    tooltip=t["theme"],
                    on_click=toggle_theme,
                    icon_size=20,
                    style=ft.ButtonStyle(
                        overlay_color={
                            ft.ControlState.HOVERED: ft.Colors.with_opacity(
                                0.1, ft.Colors.ON_SURFACE
                            ),
                            ft.ControlState.PRESSED: ft.Colors.with_opacity(
                                0.2, ft.Colors.ON_SURFACE
                            ),
                        }
                    ),
                ),
                padding=ft.padding.only(left=8),
            ),
            ft.Container(
                content=ft.Text(
                    t["title"],
                    size=18,
                    weight=ft.FontWeight.W_500,
                    color=ft.Colors.ON_SURFACE,
                    text_align=ft.TextAlign.CENTER,
                ),
                expand=True,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Row(window_controls, spacing=2),
                padding=ft.padding.only(right=8),
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    container = ft.Container(
        content=(
            row_content
            if page.web
            else ft.WindowDragArea(content=row_content, expand=True)
        ),
        height=48,
        bgcolor=ft.Colors.SURFACE,
        border=ft.border.only(
            bottom=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)
        ),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=2,
            color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
            offset=ft.Offset(0, 1),
        ),
        padding=0,
        margin=ft.margin.only(),
        expand=True,
    )

    return container

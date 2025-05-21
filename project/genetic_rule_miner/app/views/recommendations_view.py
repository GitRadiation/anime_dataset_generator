import flet as ft
from genetic_rule_miner.app.components.top_app_bar import top_app_bar


def recommendations_view(selected_series, page):
    def go_back(e):
        page.go("/search_series")

    return ft.Column(
        [
            top_app_bar(page),
            ft.Text("Recommendations", size=24, weight=ft.FontWeight.BOLD),
            ft.ResponsiveRow(
                [
                    ft.Container(
                        content=ft.Text(
                            f"Selected series: {', '.join(selected_series)}"
                        ),
                        col={"xs": 12, "sm": 8, "md": 6},
                    )
                ]
            ),
            ft.ElevatedButton(
                "Back", icon=ft.Icons.ARROW_BACK, on_click=go_back
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

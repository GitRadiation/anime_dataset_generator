import flet as ft
import requests
from genetic_rule_miner.app.components.top_app_bar import top_app_bar


def user_profile_view(page: ft.Page):
    username_field = ft.TextField(label="Enter your username", width=300)
    user_info_data_table = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("Field")),
            ft.DataColumn(label=ft.Text("Value")),
        ],
        rows=[],
    )
    user_info_table = ft.Container(
        content=user_info_data_table,
        height=300,  # Ajusta el alto máximo
        expand=False,
    )

    series_data_table = ft.DataTable(
        columns=[ft.DataColumn(label=ft.Text("Series"))], rows=[]
    )
    series_table = ft.Container(
        content=series_data_table,
        height=300,  # Ajusta el alto máximo
        expand=False,
    )

    error_text = ft.Text("", color=ft.Colors.RED)

    def fetch_profile(e):
        username = (username_field.value or "").strip()
        if not username:
            error_text.value = "Username cannot be empty"
            page.update()
            return

        try:
            res = requests.get(
                f"http://localhost:8000/users/{username}/full_profile"
            )
            res.raise_for_status()
            data = res.json()

            user_rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(key)),
                        ft.DataCell(ft.Text(str(value))),
                    ]
                )
                for key, value in data["profile"].items()
            ]

            series_rows = [
                ft.DataRow(cells=[ft.DataCell(ft.Text(serie))])
                for serie in data.get("series", [])
            ]

            user_info_data_table.rows = user_rows
            series_data_table.rows = series_rows

            error_text.value = ""

        except requests.RequestException as ex:
            error_text.value = f"Error fetching user profile: {ex}"

        page.update()

    def go_back(e):
        page.go("/")

    return ft.Column(
        [
            top_app_bar(page),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "User Profile", size=24, weight=ft.FontWeight.BOLD
                        ),
                        username_field,
                        ft.ElevatedButton(
                            "Load Profile",
                            icon=ft.Icons.SEARCH,
                            on_click=fetch_profile,
                        ),
                        error_text,
                        ft.Text(
                            "User Information",
                            size=20,
                            weight=ft.FontWeight.W_600,
                        ),
                        user_info_table,
                        ft.Text(
                            "Favorite Series",
                            size=20,
                            weight=ft.FontWeight.W_600,
                        ),
                        series_table,
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    alignment=ft.MainAxisAlignment.START,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            ),
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

import flet as ft
import requests

from genetic_rule_miner.app.components.top_app_bar import top_app_bar


def user_profile_view(page: ft.Page):
    username_field = ft.TextField(label="Enter your username", width=300)

    series_data_table = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("Series Name")),
            ft.DataColumn(label=ft.Text("Count")),
        ],
        rows=[],
    )
    series_table = ft.Container(
        content=series_data_table,
        height=300,
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
                f"http://localhost:8000/users/{username}/recomendation"
            )
            res.raise_for_status()
            data = res.json()
            error_text.value = ""

            # Limpiar tablas previas
            if series_data_table.rows:
                series_data_table.rows.clear()

            # Mostrar series recomendadas
            for item in data:
                name = item.get("nombre", "Unknown")
                cantidad = item.get("cantidad", 0)
                series_data_table.rows.append(  # type: ignore
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(name)),
                            ft.DataCell(ft.Text(str(cantidad))),
                        ]
                    )
                )

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

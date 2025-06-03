import os
import webbrowser

import flet as ft
import requests
from dotenv import load_dotenv

from genetic_rule_miner.app.components.top_app_bar import top_app_bar


def user_profile_view(page: ft.Page):
    def fetch_profile(e):
        load_dotenv(".local.env")
        api_url = os.getenv("API_URL")

        username = (username_field.value or "").strip()
        if not username:
            error_text.value = "Username cannot be empty"
            page.update()
            return

        try:
            res = requests.get(f"{api_url}/users/{username}/recommendation")
            res.raise_for_status()
            data = res.json()
            error_text.value = ""

            # Limpiar controles previos
            series_data_list.controls.clear()

            # Agregar hasta 5 elementos con mejor diseño
            for i, item in enumerate(data[:5], 1):
                name = item.get("nombre", "Unknown")
                # Crear una tarjeta para cada serie
                series_card = ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(
                                    str(i),
                                    size=18,
                                    weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.WHITE,
                                ),
                                width=40,
                                height=40,
                                bgcolor=ft.Colors.PURPLE_400,
                                border_radius=20,
                                alignment=ft.alignment.center,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        name,
                                        size=16,
                                        weight=ft.FontWeight.W_500,
                                    ),
                                    ft.Text(
                                        "Recommended for you",
                                        size=12,
                                    ),
                                ],
                                spacing=2,
                                expand=True,
                            ),
                            ft.Icon(
                                ft.Icons.PLAY_ARROW_ROUNDED,
                                color=ft.Colors.PURPLE_400,
                                size=24,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=15,
                    ),
                    padding=ft.padding.all(16),
                    margin=ft.margin.only(bottom=8),
                    border_radius=12,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    ink=True,
                    on_click=lambda e, anime_id=item.get(
                        "anime_id"
                    ): webbrowser.open(
                        f"https://myanimelist.net/anime/{anime_id}"
                    ),
                )
                series_data_list.controls.append(series_card)

        except requests.RequestException as ex:
            if ex.response is not None:
                status = ex.response.status_code
                reason = ex.response.reason
                response_text = f"{status} - {reason}"
            else:
                response_text = str(ex)

            error_text.value = f"Error fetching user profile: {response_text}"

        page.update()

    username_field = ft.TextField(
        label="Enter your MyAnimeList username",
        on_submit=fetch_profile,
        width=300,
        border_radius=10,
        filled=True,
        prefix_icon=ft.Icons.PERSON,
    )

    # Contenedor mejorado para las series
    series_data_list = ft.Column(
        spacing=0,
        scroll=ft.ScrollMode.AUTO,
        height=400,
    )

    error_text = ft.Text("", color=ft.Colors.RED)

    # Contenedor con título mejorado para las series recomendadas
    recommended_section = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.RECOMMEND,
                                color=ft.Colors.PURPLE_400,
                                size=24,
                            ),
                            ft.Text(
                                "Recommended Series",
                                size=20,
                                weight=ft.FontWeight.W_600,
                                color=ft.Colors.ON_SURFACE,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                    margin=ft.margin.only(bottom=16),
                ),
                ft.Container(
                    content=series_data_list,
                    bgcolor=ft.Colors.SURFACE,
                    border_radius=16,
                    padding=ft.padding.all(16),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    width=500,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        margin=ft.margin.only(top=20),
        alignment=ft.alignment.center,
    )

    return ft.Column(
        [
            top_app_bar(page),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Container(
                            content=ft.Text(
                                "User Profile",
                                size=28,
                                weight=ft.FontWeight.BOLD,
                            ),
                            margin=ft.margin.only(bottom=20),
                        ),
                        ft.Container(
                            content=ft.Column(
                                [
                                    username_field,
                                    ft.ElevatedButton(
                                        "Load Profile",
                                        icon=ft.Icons.SEARCH,
                                        on_click=fetch_profile,
                                        style=ft.ButtonStyle(
                                            bgcolor=ft.Colors.PURPLE_400,
                                            color=ft.Colors.WHITE,
                                            elevation=4,
                                        ),
                                    ),
                                    error_text,
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                spacing=16,
                            ),
                            margin=ft.margin.only(bottom=20),
                        ),
                        recommended_section,
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    alignment=ft.MainAxisAlignment.START,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.padding.all(20),
            ),
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

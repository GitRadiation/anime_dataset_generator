import logging
import os
import time
import webbrowser

import flet as ft
import requests
from dotenv import load_dotenv

from genetic_rule_miner.app.components.top_app_bar import top_app_bar
from genetic_rule_miner.app.utils.theme import setup_theme

logger = logging.getLogger(__name__)


def main(page: ft.Page):
    # Configurar el tamaño mínimo de la ventana
    page.window.min_width = 600
    page.window.min_height = 950
    page.window.height = page.window.min_height
    page.window.width = page.window.min_width
    page.window.title_bar_hidden = True  # Ocultar la barra de título nativa
    setup_theme(page)
    # Variables globales para los componentes
    loading_ring = ft.ProgressRing(width=40, height=40, visible=False)

    def fetch_recommended_anime_fallback(max_retries=5, timeout=10):
        """Obtiene recomendaciones de anime desde la API de Jikan como fallback"""
        url = "https://api.jikan.moe/v4/recommendations/anime"
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Attempt {attempt + 1} to fetch fallback recommendations"
                )
                response = requests.get(url, timeout=timeout)
                if response.status_code == 404:
                    logger.warning(
                        "Fallback API returned 404. Stopping retries."
                    )
                    return None
                response.raise_for_status()
                data = response.json()
                # Verificar que la respuesta tenga datos válidos
                if data and "data" in data and data["data"]:
                    return data
                else:
                    logger.warning("Fallback API returned empty data")
                    return None
            except requests.Timeout:
                logger.warning(f"Attempt {attempt + 1} timed out")
            except requests.ConnectionError:
                logger.warning(
                    f"Attempt {attempt + 1} failed: Connection error"
                )
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = min(2**attempt, 10)  # Máximo 10 segundos
                logger.debug(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
        logger.error(
            f"Failed to fetch fallback recommendations after {max_retries} attempts"
        )
        return None

    def show_no_recommendations_message():
        """Muestra un mensaje informativo cuando no hay recomendaciones"""
        no_data_container = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(
                        ft.Icons.SEARCH_OFF,
                        size=48,
                        color=ft.Colors.GREY_400,
                    ),
                    ft.Text(
                        "No recommended load found",
                        size=18,
                        weight=ft.FontWeight.W_500,
                        color=ft.Colors.GREY_600,
                    ),
                    ft.Text(
                        "Try with a different username or check back later.",
                        size=14,
                        color=ft.Colors.GREY_500,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
            ),
            padding=ft.padding.all(40),
            alignment=ft.alignment.center,
        )
        series_data_list.controls.append(no_data_container)

    def fetch_profile(e):
        load_dotenv(".local.env")
        api_url = os.getenv("API_URL")
        username = (username_field.value or "").strip()

        if not username:
            error_text.value = "Username cannot be empty"
            error_text.color = ft.Colors.RED
            page.update()
            return

        # Limpiar estados anteriores
        error_text.value = ""
        series_data_list.controls.clear()
        loading_ring.visible = True
        page.update()

        try:
            # Intentar obtener recomendaciones de la API principal
            logger.info(f"Fetching recommendations for user: {username}")
            response = requests.get(
                f"{api_url}/users/{username}/recommendation", timeout=60
            )
            response.raise_for_status()
            data = response.json()
            show_message = False
            # Si no hay datos de la API principal, intentar fallback
            if not data or (isinstance(data, list) and len(data) == 0):
                show_message = True
            series_data_list.controls.clear()
            if isinstance(data, list) and len(data) > 0:
                for i, item in enumerate(data[:5], 1):
                    name = item.get("nombre", "Unknown Title")
                    anime_id = item.get("anime_id")

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
                                            max_lines=2,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                        ),
                                        ft.Text(
                                            "Recommended for you",
                                            size=12,
                                            color=ft.Colors.GREY_600,
                                        ),
                                    ],
                                    spacing=2,
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.OPEN_IN_NEW,
                                    icon_color=ft.Colors.PURPLE_400,
                                    icon_size=20,
                                    tooltip="Open in MyAnimeList",
                                    on_click=lambda e, anime_id=anime_id: (
                                        webbrowser.open(
                                            f"https://myanimelist.net/anime/{anime_id}"
                                        )
                                        if anime_id
                                        else None
                                    ),
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
                        on_hover=lambda e: setattr(
                            e.control,
                            "bgcolor",
                            (
                                ft.Colors.ON_SURFACE_VARIANT
                                if not e.data
                                else ft.Colors.SURFACE_CONTAINER_HIGHEST
                            ),
                        ),
                    )
                    series_data_list.controls.append(series_card)
                error_text.value = ""

            if show_message:
                show_no_recommendations_message()
        except requests.Timeout:
            error_text.value = "Request timed out. Please try again."
            error_text.color = ft.Colors.RED
        except requests.ConnectionError:
            error_text.value = (
                "Connection error. Check your internet connection."
            )
            error_text.color = ft.Colors.RED
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                error_text.value = f"User '{username}' not found."
            elif e.response.status_code == 429:
                error_text.value = "Too many requests. Please wait a moment."
            else:
                error_text.value = f"Server error: {e.response.status_code}"
            error_text.color = ft.Colors.RED
        except requests.RequestException as e:
            error_text.value = f"Network error: {str(e)}"
            error_text.color = ft.Colors.RED
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            error_text.value = (
                "An unexpected error occurred. Please try again."
            )
            error_text.color = ft.Colors.RED
        finally:
            loading_ring.visible = False
            page.update()

    # Componentes de la interfaz
    username_field = ft.TextField(
        label="Enter your MyAnimeList username",
        hint_text="e.g., your_username",
        on_submit=fetch_profile,
        width=300,
        border_radius=10,
        filled=True,
        prefix_icon=ft.Icons.PERSON,
        label_style=(ft.TextStyle(color=ft.Colors.SECONDARY)),
        autofocus=True,
    )
    series_data_list = ft.Column(
        spacing=0,
        scroll=ft.ScrollMode.AUTO,
        expand=True,  # Aquí está la clave
    )

    error_text = ft.Text("", size=14)

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
                    padding=ft.padding.all(20),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    width=650,
                    height=465,
                ),
                ft.Container(
                    content=loading_ring,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(top=10),
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        margin=ft.margin.only(top=20),
        alignment=ft.alignment.center,
        expand=True,
    )

    # Contenido scrollable (todo excepto el top bar)
    scrollable_content = ft.Container(
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
            height=page.height - 48,  # type: ignore
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.padding.all(20),
        expand=True,  # Ocupa todo el espacio disponible
    )

    # Layout principal con top bar fijo y contenido scrollable
    main_content = ft.Column(
        [
            top_app_bar(page),  # Top bar fijo
            ft.Container(
                content=scrollable_content,
                expand=True,  # El contenido scrollable ocupa el resto del espacio
            ),
        ],
        spacing=0,
        expand=True,
    )

    # Crear vista con scroll solo en el contenido principal
    scrollable_view = ft.Column(
        [main_content],
        scroll=ft.ScrollMode.AUTO,  # Scroll solo en el contenido
        expand=True,
    )

    page.add(scrollable_view)


if __name__ == "__main__":
    from flet import app

    app(target=main)

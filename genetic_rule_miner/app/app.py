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
    page.window.min_width = 600
    page.window.min_height = 950
    page.window.height = page.window.min_height
    page.window.width = page.window.min_width
    page.window.title_bar_hidden = True  # Ocultar la barra de título nativa
    app_bar_container = ft.Container()
    setup_theme(page)

    # Inicializar idioma por sesión
    if not page.session.get("lang"):
        page.session.set("lang", "en")

    translations = {
        "en": {
            "enter_username": "Enter your MyAnimeList username",
            "hint_username": "e.g., your_username",
            "load_profile": "Load Profile",
            "recommended_series": "Recommended Series",
            "recommended_for_you": "Recommended for you",
            "no_recommendations": "No recommended load found",
            "try_later": "Try with a different username or check back later.",
            "language": "Language",
            "title": "Series Recommender",
            "theme": "Toggle theme",
            "minimize": "Minimize",
            "maximize": "Maximize/Restore",
            "close": "Close",
        },
        "es": {
            "enter_username": "Introduce tu usuario de MyAnimeList",
            "hint_username": "ej., tu_usuario",
            "load_profile": "Cargar Perfil",
            "recommended_series": "Series Recomendadas",
            "recommended_for_you": "Recomendado para ti",
            "no_recommendations": "No se encontraron recomendaciones",
            "try_later": "Prueba con otro usuario o vuelve más tarde.",
            "language": "Idioma",
            "title": "Recomendador de Series",
            "theme": "Cambiar tema",
            "minimize": "Minimizar",
            "maximize": "Maximizar/Restaurar",
            "close": "Cerrar",
        },
    }

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
        lang = page.session.get("lang") or "en"
        t = translations.get(lang, translations["en"])

        no_data_container = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(
                        ft.Icons.SEARCH_OFF,
                        size=48,
                        color=ft.Colors.GREY_400,
                    ),
                    ft.Text(
                        t["no_recommendations"],
                        size=18,
                        weight=ft.FontWeight.W_500,
                        color=ft.Colors.GREY_600,
                    ),
                    ft.Text(
                        t["try_later"],
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
        api_url = os.getenv("API_URL", "http://uvicorn_server:8000")
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
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=2,
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.OPEN_IN_NEW,
                                    icon_color=ft.Colors.PURPLE_400,
                                    icon_size=20,
                                    on_click=(
                                        (
                                            lambda e, anime_id=anime_id: webbrowser.open(
                                                f"https://myanimelist.net/anime/{anime_id}"
                                            )
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
                        bgcolor=(ft.Colors.SURFACE),
                        on_click=(
                            (
                                lambda e, anime_id=anime_id: webbrowser.open(
                                    f"https://myanimelist.net/anime/{anime_id}"
                                )
                            )
                            if anime_id
                            else logger.info("No hay anime_id")
                        ),
                        tooltip="Open in MyAnimeList",
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

    lang = page.session.get("lang") or "en"
    t = translations.get(lang, translations["en"])

    load_button = ft.ElevatedButton(
        t["load_profile"],
        icon=ft.Icons.SEARCH,
        on_click=fetch_profile,
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.PRIMARY,
            color=ft.Colors.WHITE,
            elevation=4,
        ),
    )

    recommended_title = ft.Text(
        t["recommended_series"],
        size=20,
        weight=ft.FontWeight.W_600,
        color=ft.Colors.ON_SURFACE,
    )

    def update_labels():
        lang = page.session.get("lang")
        t = translations.get(lang or "en", translations["en"])
        username_field.label = t["enter_username"]
        username_field.hint_text = t["hint_username"]
        load_button.text = t["load_profile"]
        recommended_title.value = t["recommended_series"]
        language_selector.tooltip = t["language"]
        app_bar_container.content = top_app_bar(page, translations)
        page.update()

    def change_language(e):
        page.session.set("lang", e.control.data)
        update_labels()

    lang = page.session.get("lang") or "en"

    language_selector = ft.PopupMenuButton(
        items=[
            ft.PopupMenuItem(
                text="English", data="en", on_click=change_language
            ),
            ft.PopupMenuItem(
                text="Español", data="es", on_click=change_language
            ),
        ],
        icon=ft.Icons.LANGUAGE,
        tooltip=translations[lang]["language"],
    )

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
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
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
                                color=ft.Colors.PRIMARY,
                                size=24,
                            ),
                            recommended_title,
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
                    padding=ft.padding.only(
                        top=20, left=20, right=20, bottom=60
                    ),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    width=650,
                    expand=True,
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

    # Contenido scrollable
    scrollable_content = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=ft.Row(
                        [language_selector],
                        alignment=ft.MainAxisAlignment.END,
                    ),
                    margin=ft.margin.only(bottom=20),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            username_field,
                            load_button,
                            error_text,
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=16,
                    ),
                    margin=ft.margin.only(bottom=20),
                ),
                recommended_section,
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.padding.all(20),
        expand=True,
    )
    app_bar_container.content = top_app_bar(page, translations)

    # Layout principal con top bar fijo y contenido scrollable
    main_content = ft.Column(
        [
            app_bar_container,
            ft.Container(
                content=scrollable_content,
                expand=True,
            ),
        ],
        spacing=0,
        expand=True,
    )

    # Crear vista con scroll solo en el contenido principal
    scrollable_view = ft.Column(
        [main_content],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    page.add(scrollable_view)
    update_labels()


app = ft.app(
    target=main, export_asgi_app=True
)  # export_asgi_app=False para poder ejecutarlo como app de escritorio

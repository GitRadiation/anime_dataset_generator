from flet import Page, View

from genetic_rule_miner.app.utils.theme import setup_theme
from genetic_rule_miner.app.views.user_profile_view import user_profile_view


def main(page: Page):
    # Configurar el tamaño mínimo de la ventana
    page.window.min_width = 600
    page.window.min_height = 900
    page.window.height = page.window.min_height
    page.window.width = page.window.min_width
    page.views.append(View("/", controls=[user_profile_view(page)]))
    page.window.title_bar_hidden = True  # Ocultar la barra de título nativa
    page.window.title_bar_buttons_hidden = True  # Ocultar botones de ventana
    setup_theme(page)
    page.go("/")


if __name__ == "__main__":
    from flet import app

    app(target=main)

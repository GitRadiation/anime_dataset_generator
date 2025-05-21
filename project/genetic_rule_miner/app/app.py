from flet import Page
from genetic_rule_miner.app.utils.theme import setup_theme
from genetic_rule_miner.app.views.recommendations_view import (
    recommendations_view,
)
from genetic_rule_miner.app.views.user_profile_view import user_profile_view


def main(page: Page):
    # Configurar el tamaño mínimo de la ventana
    page.window.min_width = 600
    page.window.min_height = 700

    setup_theme(page)
    configure_routes(page)
    page.go("/")


def configure_routes(page: Page):
    page.on_route_change = lambda route: handle_route_change(page, route)
    page.on_view_pop = lambda view: handle_view_pop(page)


def handle_route_change(page: Page, route):
    page.views.clear()
    from flet import View

    if route.route == "/":
        page.views.append(View("/", controls=[user_profile_view(page)]))
    elif route.route == "/recommendations":
        from flet import View

        page.views.append(
            View(
                "/recommendations",
                controls=[
                    recommendations_view(
                        getattr(page, "selected_series", []), page
                    )
                ],
            )
        )
    page.update()


def handle_view_pop(page: Page):
    page.views.pop()
    if len(page.views) == 0:
        page.go("/")


if __name__ == "__main__":
    from flet import app

    app(target=main)

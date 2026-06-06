import flet as ft
import os
from ui.loading_screen import show_loading_screen
from ui.chat_view import create_chat_view


async def main(page: ft.Page):
    cwd = os.getcwd()
    page.title = "AI Чат"
    page.window.icon = f"{cwd}/images/icon.ico"
    await show_loading_screen(page)
    page.controls.clear()
    page.add(await create_chat_view(page))
    await page.window.center()
    page.update()


if __name__ == "__main__":
    ft.run(main=main)
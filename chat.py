# chat.py (обновлённый)
import flet as ft
import asyncio
import threading
import os
from modules.ai import load_model, get_pipe, gen_message

async def main(page: ft.Page):
    cwd = os.getcwd()
    page.window.width = 300
    page.window.height = 200
    page.title = "AI Чат"
    page.window.icon = f"{cwd}/images/icon.ico"
    await page.window.center()

    page.theme = ft.Theme(color_scheme=ft.ColorScheme(primary=ft.Colors.RED,secondary=ft.Colors.WHITE))

    # Состояние приложения
    model_loaded = False

    def create_loading_view():
        status = ft.Text("Загрузка модели", size=20, weight=ft.FontWeight.BOLD)
        progress = ft.ProgressBar(width=300,color=ft.Colors.WHITE)
        dots = ft.Text("", size=18)
        return ft.Column(
            [
                ft.Icon(icon=ft.Icons.DOWNLOADING, size=50, color=ft.Colors.BLUE),
                ft.Row([status,dots],alignment=ft.MainAxisAlignment.CENTER),
                progress,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

    loading_view = create_loading_view()

    async def animate_dots():
        frames = ["  ", ".  ", ".. ", "..."]
        idx = 0
        dots_text = loading_view.controls[1].controls[1]
        while not model_loaded:
            dots_text.value = frames[idx % len(frames)]
            idx += 1
            page.update()
            await asyncio.sleep(0.3)

    # Показываем экран загрузки
    page.add(loading_view)
    page.update()
    animation_task = asyncio.create_task(animate_dots())

    # Фоновая загрузка модели
    def load_model_background():
        nonlocal model_loaded
        try:
            load_model()
            model_loaded = True
        except Exception as e:
            loading_view.controls[1].value = f"Ошибка загрузки: {e}"
            loading_view.controls[1].color = ft.Colors.RED
            loading_view.controls[2].visible = False
            page.update()

    threading.Thread(target=load_model_background, daemon=True).start()

    # Ждём, пока модель загрузится (периодическая проверка)
    while not model_loaded:
        await asyncio.sleep(0.1)
    animation_task.cancel()

    # Заменяем экран загрузки на чат
    page.controls.clear()
    page.add(await create_chat_view(page))
    page.update() 

async def create_chat_view(page: ft.Page):
    """Создаёт интерфейс чата."""
    cwd = os.getcwd()
    page.window.width = 1000
    page.window.height = 800
    page.padding = 0

    message_list = []
    chat_box = ft.Column(
        controls=message_list,
        scroll=ft.ScrollMode.AUTO,
        auto_scroll=True,
        width=1000,
        height=650,
        expand=True
    )

    async def send_message(e):
        user_text = chat_textfield.value.strip()
        if not user_text:
            return
        user_message = ft.Container(
            content=ft.Text(value=user_text, size=15),
            align=ft.Alignment.TOP_RIGHT,
            bgcolor=ft.Colors.with_opacity(0.9,"#f90807"),
            border_radius=5,
            padding=10,
        )
        message_list.append(user_message)
        chat_textfield.value = ""
        page.update()
        asyncio.create_task(_generate_bot_response(user_text, message_list, page))

    chat_textfield = ft.TextField(
        label="Введите ваше сообщение", width=920, multiline=True,bgcolor="#ebebeb",color= ft.Colors.BLACK, border_color=ft.Colors.RED,label_style=ft.TextStyle(color=ft.Colors.RED)
    )
    
    return ft.Container(content=
        ft.Column([chat_box,
            ft.Row([chat_textfield,
                    ft.IconButton(icon=ft.Icons.SEND, 
                                on_click=send_message,
                                icon_color=ft.Colors.RED,
                                hover_color=ft.Colors.WHITE,
                                bgcolor=ft.Colors.with_opacity(0.8,ft.Colors.WHITE))],)
        ],
            expand=True,
            margin=10
        ),
        image=ft.DecorationImage(src=f"{cwd}/images/bg4.jpg",fit=ft.BoxFit.COVER),
        expand=True
    )

async def _generate_bot_response(user_text: str, message_list: list, page: ft.Page):
    # Индикатор набора текста
    typing_indicator = ft.Container(
        content=ft.Text(value=".", size=15, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.with_opacity(0.85,ft.Colors.RED_ACCENT_700),
        align=ft.Alignment.CENTER_LEFT,
        border_radius=15,
        padding=ft.Padding.symmetric(horizontal=20, vertical=10),
        width=80,
        height=45,
    )
    message_list.append(typing_indicator)
    page.update()

    animation_task = asyncio.create_task(_animate_typing(typing_indicator, page))

    try:
        loop = asyncio.get_running_loop()
        bot_answer = await loop.run_in_executor(None, gen_message, user_text)
        bot_answer = str(bot_answer).strip()
    except Exception as ex:
        bot_answer = f"Ошибка: {ex}"
    finally:
        animation_task.cancel()
        if typing_indicator in message_list:
            message_list.remove(typing_indicator)

    bot_message = ft.Container(
        content=ft.Markdown(
            value=bot_answer,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            auto_follow_links=True,
            
        ),
        bgcolor=ft.Colors.with_opacity(0.85,ft.Colors.RED_ACCENT_700),
        align=ft.Alignment.TOP_LEFT,
        border_radius=5,
        padding=10,
    )
    message_list.append(bot_message)
    page.update()

async def _animate_typing(indicator: ft.Container, page: ft.Page):
    dots = [".", "..", "..."]
    idx = 0
    try:
        while True:
            indicator.content.value = dots[idx % len(dots)]
            idx += 1
            page.update()
            await asyncio.sleep(0.4)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    ft.run(main=main)
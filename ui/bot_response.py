
import asyncio
import flet as ft

from modules.ai import generate_response as gen_message


# ---------------------------------------------------------
# Карта статусов модели
# ---------------------------------------------------------

_STATUS_MAP = {
    "ищу в интернете": (
        "🔍",
        ft.Colors.BLUE_200,
    ),
    "анализирую результаты": (
        "📊",
        ft.Colors.PURPLE_200,
    ),
    "получаю текущее время": (
        "🕐",
        ft.Colors.TEAL_200,
    ),
    "думаю над ответом": (
        "💭",
        ft.Colors.AMBER_200,
    ),
    "формирую ответ": (
        "✍️",
        ft.Colors.GREEN_200,
    ),
}


# ---------------------------------------------------------
# Преобразование статуса в иконку
# ---------------------------------------------------------

def status_to_icon_color(status: str):
    status_lower = status.lower()

    for key, (icon, color) in _STATUS_MAP.items():
        if key in status_lower:
            return icon, color

    return "⚙️", ft.Colors.WHITE70


# ---------------------------------------------------------
# Генерация ответа
# ---------------------------------------------------------

async def generate_bot_response(
    user_text: str,
    message_list: list,
    page: ft.Page,
):
    """
    Генерирует ответ модели
    и отображает его в интерфейсе.
    """

    # -----------------------------------------------------
    # Индикатор выполнения
    # -----------------------------------------------------

    status_icon = ft.Text(
        "⚙️",
        size=18,
    )

    status_label = ft.Text(
        "Обрабатываю запрос...",
        size=13,
        color=ft.Colors.WHITE,
        weight=ft.FontWeight.W_500,
    )

    indicator = ft.Container(
        content=ft.Row(
            [
                status_icon,
                status_label,
            ],
            spacing=8,
            tight=True,
            vertical_alignment=(
                ft.CrossAxisAlignment.CENTER
            ),
        ),
        bgcolor=ft.Colors.with_opacity(
            0.85,
            ft.Colors.RED_ACCENT_700,
        ),
        border_radius=20,
        padding=16,
    )

    message_list.append(indicator)

    page.update()

    # -----------------------------------------------------
    # Колбэк обновления статуса
    # -----------------------------------------------------

    def on_status(message: str):
        icon, color = status_to_icon_color(
            message
        )

        status_icon.value = icon
        status_label.value = message
        status_label.color = color

        page.update()

    # -----------------------------------------------------
    # Анимация индикатора
    # -----------------------------------------------------

    pulse_running = True

    async def pulse_animation():
        state = True

        while pulse_running:
            status_icon.opacity = (
                1.0 if state else 0.4
            )

            page.update()

            state = not state

            await asyncio.sleep(0.6)

    pulse_task = asyncio.create_task(
        pulse_animation()
    )

    # -----------------------------------------------------
    # Запуск модели
    # -----------------------------------------------------

    try:
        loop = asyncio.get_running_loop()

        bot_answer = await loop.run_in_executor(
            None,
            lambda: gen_message(
                user_text,
                status_callback=on_status,
            ),
        )

        bot_answer = str(
            bot_answer
        ).strip()

    except Exception as ex:
        bot_answer = (
            f"Ошибка: {ex}"
        )

    finally:
        pulse_running = False

        pulse_task.cancel()

        if indicator in message_list:
            message_list.remove(
                indicator
            )

    # -----------------------------------------------------
    # Отображение ответа
    # -----------------------------------------------------

    message_list.append(
        ft.Container(
            content=ft.Markdown(
                value=bot_answer,
                selectable=True,
                extension_set=(
                    ft.MarkdownExtensionSet
                    .GITHUB_WEB
                ),
                auto_follow_links=True,
            ),
            bgcolor=ft.Colors.with_opacity(
                0.85,
                ft.Colors.RED_ACCENT_700,
            ),
            align=ft.Alignment.TOP_LEFT,
            border_radius=5,
            padding=10,
        )
    )

    page.update()


import asyncio
import threading
import flet as ft

from modules.ai import load_model

# ─────────────────────────────────────────────────────────────
# Этапы загрузки модели
# ─────────────────────────────────────────────────────────────

LOAD_STAGES = [
    (0.05, "Подготовка конфигурации..."),
    (0.15, "Загрузка токенизатора..."),
    (0.30, "Загрузка весов модели..."),
    (0.90, "Перенос модели на устройство..."),
    (1.00, "Готово!"),
]


async def show_loading_screen(page: ft.Page):
    """
    Отображает экран загрузки модели.
    Завершается только после успешной загрузки модели.
    """

    model_loaded = False
    load_error = None

    # ---------------------------------------------------------
    # Настройка окна
    # ---------------------------------------------------------

    page.window.width = 420
    page.window.height = 280

    page.bgcolor = "#1a1a1a"

    # ---------------------------------------------------------
    # Элементы интерфейса
    # ---------------------------------------------------------

    stage_text = ft.Text(
        "Инициализация...",
        size=14,
        color=ft.Colors.WHITE70,
    )

    percent_text = ft.Text(
        "0%",
        size=13,
        color=ft.Colors.WHITE54,
        weight=ft.FontWeight.BOLD,
    )

    progress_bar = ft.ProgressBar(
        width=380,
        value=0.0,
        color=ft.Colors.RED,
        bgcolor=ft.Colors.WHITE24,
    )

    dots_text = ft.Text(
        "",
        size=16,
        color=ft.Colors.WHITE,
    )

    loading_view = ft.Column(
        [
            ft.Icon(
                icon=ft.Icons.MEMORY,
                size=52,
                color=ft.Colors.RED,
            ),
            ft.Text(
                "Загрузка модели",
                size=22,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.WHITE,
            ),
            ft.Row(
                [stage_text, dots_text],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=2,
            ),
            ft.Container(height=8),
            progress_bar,
            ft.Container(height=4),
            percent_text,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=6,
    )

    page.controls.clear()

    page.add(
        ft.Container(
            content=loading_view,
            alignment=ft.Alignment.CENTER,
            expand=True,
            padding=30,
        )
    )

    page.update()

    # ---------------------------------------------------------
    # Анимация точек
    # ---------------------------------------------------------

    async def animate_dots():
        frames = ["", ".", "..", "..."]
        idx = 0

        while not model_loaded and not load_error:
            dots_text.value = frames[idx % len(frames)]
            idx += 1

            page.update()

            await asyncio.sleep(0.4)

    animation_task = asyncio.create_task(
        animate_dots()
    )

    # ---------------------------------------------------------
    # Колбэк прогресса
    # ---------------------------------------------------------

    def on_progress(message: str, percent: float):
        stage_text.value = message
        percent_text.value = f"{int(percent * 100)}%"
        progress_bar.value = percent

        page.update()

    # ---------------------------------------------------------
    # Загрузка модели в отдельном потоке
    # ---------------------------------------------------------

    def load_model_background():
        nonlocal model_loaded, load_error

        try:
            load_model(
                progress_callback=on_progress
            )

            model_loaded = True

        except Exception as e:
            load_error = str(e)

            stage_text.value = (
                f"Ошибка загрузки: {e}"
            )

            stage_text.color = ft.Colors.RED_300
            progress_bar.color = ft.Colors.RED_900

            page.update()

    threading.Thread(
        target=load_model_background,
        daemon=True,
    ).start()

    # ---------------------------------------------------------
    # Ожидание завершения загрузки
    # ---------------------------------------------------------

    while not model_loaded and not load_error:
        await asyncio.sleep(0.1)

    animation_task.cancel()

    if load_error:
        raise RuntimeError(load_error)

    # ---------------------------------------------------------
    # Очистка экрана загрузки
    # ---------------------------------------------------------

    page.bgcolor = None
    page.controls.clear()
    page.update()

    return True


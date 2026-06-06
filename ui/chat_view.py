
import os
import asyncio
import flet as ft

from modules.file_reader import read_file_content
from ui.bot_response import generate_bot_response


async def create_chat_view(page: ft.Page):
    cwd = os.getcwd()
    
    page.window.width = 1000
    page.window.height = 800
    page.padding = 0

    # ---------------------------------------------------------
    # Состояние чата
    # ---------------------------------------------------------

    message_list = []

    attached_file = {
        "path": None,
        "name": None,
        "content": None,
    }

    # ---------------------------------------------------------
    # Бейдж прикрепленного файла
    # ---------------------------------------------------------

    file_badge = ft.Container(
        content=ft.Row(
            [
                ft.Icon(
                    ft.Icons.ATTACH_FILE,
                    size=14,
                    color=ft.Colors.WHITE,
                ),
                ft.Text(
                    "",
                    size=12,
                    color=ft.Colors.WHITE,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=14,
                    icon_color=ft.Colors.WHITE,
                    tooltip="Открепить файл",
                    padding=0,
                    style=ft.ButtonStyle(
                        overlay_color=ft.Colors.TRANSPARENT
                    ),
                ),
            ],
            spacing=4,
            tight=True,
        ),
        bgcolor=ft.Colors.with_opacity(
            0.8,
            ft.Colors.RED_700,
        ),
        border_radius=8,
        padding=10,
        visible=False,
        margin=4,
    )

    # ---------------------------------------------------------
    # Область сообщений
    # ---------------------------------------------------------

    chat_box = ft.Column(
        controls=message_list,
        scroll=ft.ScrollMode.AUTO,
        auto_scroll=True,
        expand=True,
        width=1000,
        height=650,
    )

    # ---------------------------------------------------------
    # Открепление файла
    # ---------------------------------------------------------

    def detach_file():
        attached_file.update(
            {
                "path": None,
                "name": None,
                "content": None,
            }
        )

        file_badge.visible = False
        file_badge.content.controls[1].value = ""

        page.update()

    file_badge.content.controls[2].on_click = (
        lambda e: detach_file()
    )

    # ---------------------------------------------------------
    # FilePicker
    # ---------------------------------------------------------

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def pick_file(e):
        files = await file_picker.pick_files(
            dialog_title="Выберите файл",
            file_type=ft.FilePickerFileType.CUSTOM,
            allow_multiple=False,
            allowed_extensions=[
                "txt", "md", "markdown", "rst",
                "log", "csv", "tsv","json",
                "xml", "yaml", "yml", "toml", 
                "ini", "cfg", "py", "js",
                "ts", "html", "css", "java",
                "cpp", "c", "h", "cs",
                "go", "rs", "php", "rb",
                "sh", "bat", "pdf", "docx",
                "doc", "odt", "xlsx", "xls", "ods",
            ],
        )

        if not files:
            return

        selected_file = files[0]

        content, error = read_file_content(
            selected_file.path
        )

        if error:
            message_list.append(
                ft.Container(
                    content=ft.Text(
                        f"⚠️ Не удалось прочитать файл: {error}",
                        size=13,
                        color=ft.Colors.ORANGE_200,
                    ),
                    bgcolor=ft.Colors.with_opacity(
                        0.7,
                        ft.Colors.BLACK,
                    ),
                    border_radius=5,
                    padding=8,
                )
            )

            page.update()
            return

        attached_file.update(
            {
                "path": selected_file.path,
                "name": selected_file.name,
                "content": content,
            }
        )

        file_badge.content.controls[1].value = (
            selected_file.name
        )

        file_badge.visible = True

        page.update()

    # ---------------------------------------------------------
    # Поле ввода
    # ---------------------------------------------------------

    chat_textfield = ft.TextField(
        label="Введите ваше сообщение",
        width=870,
        multiline=True,
        bgcolor="#ebebeb",
        color=ft.Colors.BLACK,
        border_color=ft.Colors.RED,
        label_style=ft.TextStyle(
            color=ft.Colors.RED
        ),
    )

    # ---------------------------------------------------------
    # Отправка сообщения
    # ---------------------------------------------------------

    async def send_message(e):
        user_text = chat_textfield.value.strip()

        file_content = attached_file["content"]
        file_name = attached_file["name"]

        if not user_text and not file_content:
            return

        display_parts = []

        if file_name:
            display_parts.append(
                f"📎 **{file_name}**"
            )

        if user_text:
            display_parts.append(user_text)

        if file_content and user_text:
            model_text = (
                f"Пользователь прикрепил файл "
                f"«{file_name}».\n"
                f"Содержимое файла:\n"
                f"```\n{file_content}\n```\n\n"
                f"Вопрос пользователя: "
                f"{user_text}"
            )

        elif file_content:
            model_text = (
                f"Пользователь прикрепил файл "
                f"«{file_name}».\n"
                f"Содержимое файла:\n"
                f"```\n{file_content}\n```\n\n"
                f"Проанализируй этот файл "
                f"и кратко опиши его "
                f"содержимое."
            )

        else:
            model_text = user_text

        # сообщение пользователя

        message_list.append(
            ft.Container(
                content=ft.Markdown(
                    value="\n".join(display_parts),
                    selectable=True,
                    extension_set=(
                        ft.MarkdownExtensionSet.GITHUB_WEB
                    ),
                ),
                align=ft.Alignment.TOP_RIGHT,
                bgcolor=ft.Colors.with_opacity(
                    0.9,
                    "#f90807",
                ),
                border_radius=5,
                padding=10,
            )
        )

        chat_textfield.value = ""

        detach_file()

        page.update()

        asyncio.create_task(
            generate_bot_response(
                model_text,
                message_list,
                page,
            )
        )

    # ---------------------------------------------------------
    # Кнопки
    # ---------------------------------------------------------

    attach_button = ft.IconButton(
        icon=ft.Icons.ATTACH_FILE,
        icon_color=ft.Colors.RED,
        hover_color=ft.Colors.WHITE,
        bgcolor=ft.Colors.with_opacity(
            0.8,
            ft.Colors.WHITE,
        ),
        tooltip="Прикрепить файл",
        on_click=pick_file,
    )

    send_button = ft.IconButton(
        icon=ft.Icons.SEND,
        icon_color=ft.Colors.RED,
        hover_color=ft.Colors.WHITE,
        bgcolor=ft.Colors.with_opacity(
            0.8,
            ft.Colors.WHITE,
        ),
        on_click=send_message,
    )

    # ---------------------------------------------------------
    # Основной контейнер
    # ---------------------------------------------------------

    return ft.Container(
        image=ft.DecorationImage(
            src=f"{cwd}/images/bg4.jpg",
            fit=ft.BoxFit.COVER,
        ),
        expand=True,
        content=ft.Column(
            [
                chat_box,
                file_badge,
                ft.Row(
                    [
                        chat_textfield,
                        attach_button,
                        send_button,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                ),
            ],
            expand=True,
            margin=10,
        ),
    )


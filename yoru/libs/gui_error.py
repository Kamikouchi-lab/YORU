"""GUI 共通の例外ハンドリングヘルパー。

各 ``*_GUI`` クラスに :class:`GuiErrorMixin` を継承させることで、

- 失敗時に標準エラー出力へトレースバックを残し
  (``app.py`` の ``_run_gui_subprocess`` がホーム画面へ転送する)、
- DearPyGui のモーダルウィンドウでエラー内容を表示する

を統一的に行える。``analysis_GUI.py`` で先行実装したパターンを共通化したもの。
"""

import sys
import traceback

import dearpygui.dearpygui as dpg


class GuiErrorMixin:
    """DearPyGui ベースの GUI に共通のエラー報告機能を提供する Mixin。"""

    # 同時に複数のエラーポップアップを開かないよう固定タグを使う
    _error_popup_tag = "error_popup"

    def _report_error(self, context, exc):
        """例外を標準エラー出力に記録し、GUI上のポップアップで表示する。

        標準エラー出力へのトレースバックは、サブプロセスとして起動された
        場合に ``app.py`` 側の ``_run_gui_subprocess`` で捕捉され、ホーム
        画面に通知される。
        """
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {context}: {detail}", file=sys.stderr, flush=True)
        traceback.print_exc()
        self._show_error_popup(context, detail)

    def _show_error_popup(self, context, detail):
        """エラー内容をモーダルウィンドウで表示する。"""
        try:
            tag = self._error_popup_tag
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            with dpg.window(
                label="Error",
                modal=True,
                tag=tag,
                no_resize=True,
                width=480,
                pos=(210, 300),
            ):
                dpg.add_text(context, color=(255, 160, 120))
                dpg.add_separator()
                dpg.add_text(detail, wrap=460, color=(255, 120, 120))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Close",
                    width=80,
                    callback=lambda: dpg.delete_item(tag),
                )
        except Exception:
            # ポップアップ表示に失敗しても標準エラー出力には残っている
            pass

    def _safe_enable(self, tag):
        """存在する場合のみアイテムを有効化する(失敗しても無視)。"""
        try:
            if dpg.does_item_exist(tag):
                dpg.enable_item(tag)
        except Exception:
            pass

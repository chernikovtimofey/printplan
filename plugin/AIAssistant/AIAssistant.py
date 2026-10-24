"""STE Slicer extension: «AI Assistant» menu, local chat server, browser window.

The menu item starts the local HTTP server (chat page, chat API, MCP endpoint)
on 127.0.0.1 and opens the chat in the system browser. The LLM connection is
configured on the chat page and stored in the slicer preferences.
"""
import webbrowser

from UM.Application import Application
from UM.Extension import Extension
from UM.Logger import Logger
from UM.Message import Message

from .agent.bridge import QtMainThreadBridge
from .agent.config import PreferencesSettings


class AIAssistant(Extension):
    def __init__(self):
        super().__init__()
        self._server = None
        app = Application.getInstance()
        self._bridge = QtMainThreadBridge(app)
        self._settings = PreferencesSettings(app.getPreferences(), bridge=self._bridge, save=app.savePreferences)
        self.setMenuName("AI Assistant")
        self.addMenuItem("Открыть чат", self.showChat)
        self.addMenuItem("Адрес MCP-сервера", self.showMcpAddress)

    def _ensureServer(self):
        if self._server is None:
            from .agent.server import AssistantServer
            from .agent.slicer_live import LiveSlicer
            slicer = LiveSlicer(Application.getInstance(), self._bridge)
            self._server = AssistantServer(slicer, self._settings, log=lambda text: Logger.log("d", "[AI] %s", text))
            self._server.start()
            Logger.log("i", "AI Assistant server started at %s", self._server.url)
        return self._server

    def showChat(self):
        try:
            server = self._ensureServer()
        except Exception as e:
            Logger.logException("e", "AI Assistant server failed to start")
            Message("Не удалось запустить сервер AI Assistant: %s" % e, title="AI Assistant").show()
            return
        webbrowser.open(server.url)

    def showMcpAddress(self):
        try:
            server = self._ensureServer()
        except Exception as e:
            Logger.logException("e", "AI Assistant server failed to start")
            Message("Не удалось запустить сервер AI Assistant: %s" % e, title="AI Assistant").show()
            return
        Message("Чат: %s\nMCP (JSON-RPC, POST): %smcp" % (server.url, server.url), title="AI Assistant",
                lifetime=0).show()

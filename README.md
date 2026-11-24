# webmcp - Chrome Extension

**webmcp** is a Chrome Extension (Manifest V3) that allows you to control any **WebMCP-enabled website** using natural language and your own API key from **Gemini**, **Claude**, or **OpenAI**.

## Features

1. **Full WebMCP Standard Support**:
   - **Imperative WebMCP API** (`navigator.modelContext.registerTool`, `provideContext`, `unregisterTool`, `clearContext` and `document.modelContext`).
   - **Native Chrome WebMCP Testing Surface** (`navigator.modelContextTesting.listTools` / `executeTool`).
   - **Declarative WebMCP HTML Forms** (`<form toolname="..." tooldescription="..." toolautosubmit>` and `toolparamdescription`).
2. **Bring Your Own API Key (Powered by the Open-Source Vercel AI SDK)**:
   - Connect directly to **Google Gemini** (`@ai-sdk/google`), **Anthropic Claude** (`@ai-sdk/anthropic`), or **OpenAI** (`@ai-sdk/openai`) using the open-source `ai` SDK.
   - Keys are stored locally in your browser (`chrome.storage.local`).
3. **Interactive Prompt & Tool Inspector UI**:
   - Once your API key is saved, the extension immediately switches to the **Prompt Input UI**.
   - Automatically detects and displays all active WebMCP tools on the current webpage and executes multi-step tool calls from your natural language instructions.

## Build & Load in Chrome

1. Install dependencies and build the extension bundle into `dist/`:
   ```bash
   npm install
   npm run build
   ```
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Enable **Developer mode** (top-right toggle).
4. Click **Load unpacked** and select the `dist/` folder.
5. Open `demo/index.html` in Chrome, click the **webmcp** extension icon in the toolbar, input your Gemini / Claude / OpenAI API key, and try prompts like:
   - *"Add a todo item to buy coffee and mark the page theme light blue"*
   - *"Create a high priority support ticket for login issues"*

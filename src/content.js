/**
 * ISOLATED World Content Script
 * Bridges messages between the extension UI (sidepanel / popup) and the MAIN world page-bridge.js
 */
(function () {
  if (window.__WEBMCP_CONTENT_SCRIPT_INITIALIZED__) return;
  window.__WEBMCP_CONTENT_SCRIPT_INITIALIZED__ = true;

  const pendingRequests = new Map();
  let reqCounter = 0;

  function callPageBridge(action, payload = {}) {
    return new Promise((resolve, reject) => {
      const requestId = `webmcp_${Date.now()}_${++reqCounter}`;
      const timer = setTimeout(() => {
        pendingRequests.delete(requestId);
        reject(new Error(`WebMCP page bridge timeout for ${action}`));
      }, 15000);

      pendingRequests.set(requestId, { resolve, reject, timer });

      window.postMessage(
        {
          source: 'webmcp-content-script',
          requestId,
          action,
          payload
        },
        '*'
      );
    });
  }

  window.addEventListener('message', (event) => {
    if (event.source !== window || !event.data || event.data.source !== 'webmcp-page-bridge') {
      return;
    }

    const { type, requestId, success, data, error, tools } = event.data;

    if (type === 'WEBMCP_TOOLS_CHANGED') {
      chrome.runtime.sendMessage({
        type: 'WEBMCP_TOOLS_UPDATED',
        tools: tools || []
      }).catch(() => {});
      return;
    }

    if (requestId && pendingRequests.has(requestId)) {
      const { resolve, reject, timer } = pendingRequests.get(requestId);
      clearTimeout(timer);
      pendingRequests.delete(requestId);
      if (success) {
        resolve(data);
      } else {
        reject(new Error(error || 'Unknown WebMCP execution error'));
      }
    }
  });

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type === 'WEBMCP_GET_TOOLS') {
      callPageBridge('LIST_TOOLS')
        .then((res) => sendResponse({ ok: true, ...res }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
      return true; // asynchronous response
    }

    if (message?.type === 'WEBMCP_CALL_TOOL') {
      callPageBridge('EXECUTE_TOOL', {
        name: message.name,
        args: message.args
      })
        .then((res) => sendResponse({ ok: true, result: res }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
      return true; // asynchronous response
    }

    return false;
  });
})();

// Open side panel when the user clicks the context menu or allows both popup and side panel usage
chrome.runtime.onInstalled.addListener(() => {
  if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
    // Allow opening as sidePanel via action click if preferred, or keep default popup
    // Setting openPanelOnActionClick: true makes clicking the extension icon open the persistent side panel
    // so the user can interact with the webpage while chatting with webmcp.
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
  }
});

// Relay tool update notifications from content scripts to the UI
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === 'WEBMCP_TOOLS_UPDATED') {
    chrome.runtime.sendMessage({
      type: 'WEBMCP_TAB_TOOLS_UPDATED',
      tabId: sender.tab?.id,
      tools: message.tools
    }).catch(() => {});
  }
  return false;
});

// Toolbar badge: show the number of WebMCP tools on the current tab
function setBadge(tabId, count) {
  chrome.action.setBadgeText({ tabId, text: count > 0 ? String(count) : '' }).catch(() => {});
  chrome.action.setBadgeBackgroundColor({ tabId, color: '#1a73e8' }).catch(() => {});
}

// Runs inside the page's MAIN world; mirrors the side panel's tool discovery
async function inPageCountTools() {
  const names = new Set();
  const toArr = (raw) => {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (Array.isArray(raw.tools)) return raw.tools;
    if (raw instanceof Map) return Array.from(raw.values());
    if (typeof raw[Symbol.iterator] === 'function') return Array.from(raw);
    return [];
  };
  const sources = [
    document.modelContext,
    window.modelContext,
    navigator.modelContext,
    document.modelContextTesting,
    navigator.modelContextTesting
  ];
  for (const mc of sources) {
    if (!mc) continue;
    const fn = typeof mc.getTools === 'function' ? 'getTools' : typeof mc.listTools === 'function' ? 'listTools' : null;
    try {
      const list = fn ? toArr(await mc[fn]()) : toArr(mc.tools);
      for (const t of list) {
        const n = t?.name || t?.toolName || t?.id;
        if (n) names.add(String(n));
      }
    } catch (e) {}
  }
  document.querySelectorAll('form[toolname], [data-webmcp-tool]').forEach((f) => {
    const n = f.getAttribute('toolname') || f.getAttribute('data-webmcp-tool');
    if (n) names.add(n);
  });
  return names.size;
}

function refreshBadge(tabId) {
  chrome.scripting
    .executeScript({ target: { tabId }, world: 'MAIN', func: inPageCountTools })
    .then((results) => {
      const count = results.reduce((sum, r) => Math.max(sum, Number(r.result) || 0), 0);
      setBadge(tabId, count);
    })
    .catch(() => setBadge(tabId, 0));
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete') {
    refreshBadge(tabId);
    setTimeout(() => refreshBadge(tabId), 1000);
    setTimeout(() => refreshBadge(tabId), 3000);
  }
});
chrome.tabs.onActivated.addListener(({ tabId }) => refreshBadge(tabId));

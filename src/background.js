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

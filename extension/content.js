(() => {
  const HOST_ID = "vietnamese-asr-live-caption-host";
  let recentCaptions = [];

  function ensureOverlay() {
    let host = document.getElementById(HOST_ID);
    const player = document.querySelector("#movie_player") || document.body;
    if (host && host.parentElement !== player) host.remove();
    host = document.getElementById(HOST_ID);
    if (host) return host.shadowRoot;

    host = document.createElement("div");
    host.id = HOST_ID;
    host.setAttribute("aria-live", "polite");
    host.setAttribute("aria-atomic", "true");
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = `
      :host { position: absolute; inset: 0; z-index: 2147483646; pointer-events: none; }
      .caption-wrap { position: absolute; left: 50%; bottom: max(66px, 10%); display: flex; width: min(calc(100% - 32px), 760px); justify-content: center; transform: translateX(-50%); text-align: center; }
      .caption { display: -webkit-box; box-sizing: border-box; width: fit-content; max-width: 100%; overflow: hidden; padding: 7px 12px; border: 1px solid rgba(255, 255, 255, .14); border-radius: 12px; color: #fff; background: rgba(8, 14, 24, .82); box-shadow: 0 6px 24px rgba(0, 0, 0, .24); font: 600 clamp(16px, 1.55vw, 24px)/1.35 system-ui, sans-serif; letter-spacing: .01em; overflow-wrap: anywhere; text-wrap: pretty; text-shadow: 0 1px 2px #000; backdrop-filter: blur(8px); -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
      .caption[data-state="status"] { max-width: min(100%, 460px); padding: 6px 11px; color: #ccfbf1; font-size: clamp(14px, 1.25vw, 18px); -webkit-line-clamp: 1; }
      .caption:empty { display: none; }
    `;
    const wrap = document.createElement("div");
    wrap.className = "caption-wrap";
    const caption = document.createElement("span");
    caption.className = "caption";
    caption.setAttribute("role", "status");
    wrap.append(caption);
    shadow.append(style, wrap);
    player.append(host);
    return shadow;
  }

  function setText(text, state = "caption") {
    const shadow = ensureOverlay();
    const caption = shadow.querySelector(".caption");
    caption.dataset.state = state;
    caption.textContent = text;
  }

  function clear() {
    recentCaptions = [];
    const host = document.getElementById(HOST_ID);
    if (host?.shadowRoot) host.shadowRoot.querySelector(".caption").textContent = "";
  }

  chrome.runtime.onMessage.addListener((message) => {
    if (message.target !== "content") return;
    if (message.type === "caption") {
      recentCaptions.push(String(message.text || "").trim());
      recentCaptions = recentCaptions.filter(Boolean).slice(-2);
      setText(recentCaptions.join(" "));
    } else if (message.type === "silence") {
      clear();
    } else if (message.type === "session-starting") {
      setText("Đang kết nối phụ đề…", "status");
    } else if (message.type === "status" && message.status === "listening") {
      setText("Đang nghe âm thanh…", "status");
    } else if (message.type === "error") {
      setText(message.message || "Phụ đề gặp lỗi.", "status");
    } else if (message.type === "session-stopped" || message.type === "clear") {
      clear();
    }
  });
})();

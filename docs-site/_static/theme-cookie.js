// Share the light/dark choice with mojolearn.org through a cookie on the parent domain.
// Runs in <head> before the theme's own script: a saved cookie choice wins over this site's localStorage,
// and a light/dark pick in the theme switcher is written back to the cookie ("auto" clears it).
(function () {
  var root = document.documentElement;
  var domain = location.hostname.endsWith("mojolearn.org") ? "; Domain=mojolearn.org" : "";
  var secure = location.protocol === "https:" ? "; Secure" : "";
  var m = document.cookie.match(/(?:^|; )theme=(light|dark)/);
  if (m) {
    try { localStorage.setItem("mode", m[1]); localStorage.setItem("theme", m[1]); } catch (e) {}
    root.dataset.mode = m[1];
    root.dataset.theme = m[1];
  }
  new MutationObserver(function () {
    var mode = root.dataset.mode;
    if (mode === "light" || mode === "dark") {
      document.cookie = "theme=" + mode + "; Path=/; Max-Age=31536000; SameSite=Lax" + domain + secure;
    } else if (mode === "auto") {
      document.cookie = "theme=; Path=/; Max-Age=0; SameSite=Lax" + domain + secure;
    }
  }).observe(root, { attributes: true, attributeFilter: ["data-mode"] });
})();

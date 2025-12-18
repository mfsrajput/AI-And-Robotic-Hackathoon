// Polyfill for process.env in browser
window.process = window.process || { env: { } };
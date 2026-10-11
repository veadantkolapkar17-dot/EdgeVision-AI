/* ==========================================================================
   EdgeVision AI - Progressive Web App Service Worker
   Offline caching, app shell management, and Edge AI asset persistence
   ========================================================================== */

const CACHE_NAME = "edgevision-pwa-v1.0.0";

// Essential App Shell resources (cached immediately on install)
const PRECACHE_ASSETS = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-maskable-192.png",
  "./icons/icon-maskable-512.png",
  "./icons/icon.svg",
  "./icons/apple-touch-icon.png",
  "./icons/favicon.ico",
  "./vendor/ort.all.min.mjs",
  "./vendor/ort-wasm-simd-threaded.mjs",
  "./vendor/ort-wasm-simd-threaded.jsep.mjs"
];

// Large model & binary runtime assets (cached on install or on demand)
const OPTIONAL_PRECACHE = [
  "./models/yolo11n.onnx",
  "./vendor/ort-wasm-simd-threaded.wasm",
  "./vendor/ort-wasm-simd-threaded.jsep.wasm"
];

// Install: Cache App Shell immediately
self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      try {
        await cache.addAll(PRECACHE_ASSETS);
        console.log("[SW] App Shell pre-cached successfully");
      } catch (err) {
        console.warn("[SW] Some app shell assets could not be pre-cached:", err);
      }

      // Attempt to cache local model & runtime files in background
      for (const asset of OPTIONAL_PRECACHE) {
        try {
          const match = await cache.match(asset);
          if (!match) {
            const res = await fetch(asset);
            if (res.ok) await cache.put(asset, res);
          }
        } catch (e) {
          // Non-fatal, will be cached dynamically on first run
        }
      }

      return self.skipWaiting();
    })()
  );
});

// Activate: Clean up old caches and claim clients
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log("[SW] Removing outdated cache:", key);
            return caches.delete(key);
          }
        })
      );
      await self.clients.claim();
      console.log("[SW] Service Worker active and clients claimed");
    })()
  );
});

// Fetch: Hybrid strategy
// - App Shell (HTML/CSS/JS): Stale-While-Revalidate with network fallback
// - Models (.onnx) & WASM (.wasm): Cache-First for instant offline Edge AI
self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== "GET") return;

  // Handle CDN / Cross-origin or Large Binary Assets (ONNX & WASM)
  const isBinaryAsset =
    url.pathname.endsWith(".onnx") ||
    url.pathname.endsWith(".wasm") ||
    url.hostname.includes("jsdelivr.net") ||
    url.hostname.includes("huggingface.co");

  if (isBinaryAsset) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE_NAME);
        const cachedResponse = await cache.match(request);
        if (cachedResponse) {
          return cachedResponse;
        }

        try {
          const networkResponse = await fetch(request);
          if (networkResponse && (networkResponse.ok || networkResponse.type === "opaque")) {
            // Store a copy in cache for future offline usage
            cache.put(request, networkResponse.clone());
          }
          return networkResponse;
        } catch (fetchErr) {
          // If offline and not in cache, try checking without query params
          const cleanUrl = url.origin + url.pathname;
          const fallback = await cache.match(cleanUrl);
          if (fallback) return fallback;
          throw fetchErr;
        }
      })()
    );
    return;
  }

  // App Shell Navigation & Assets: Stale-While-Revalidate
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);

      const fetchPromise = fetch(request)
        .then((networkResponse) => {
          if (networkResponse && networkResponse.ok) {
            cache.put(request, networkResponse.clone());
          }
          return networkResponse;
        })
        .catch((err) => {
          // Offline fallback
          return cached;
        });

      return cached || fetchPromise;
    })()
  );
});

// Listen for messages from client (e.g. skipWaiting trigger, cache query)
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

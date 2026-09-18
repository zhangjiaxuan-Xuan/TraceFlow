(() => {
  'use strict';
  const config = window.TRACEFLOW_CONFIG || { links: {} };
  document.querySelectorAll('[data-resource]').forEach(link => {
    const url = (config.links?.[link.dataset.resource] || '').trim();
    if (!url) return;
    try {
      const parsed = new URL(url, location.href);
      if (!['https:', 'http:'].includes(parsed.protocol)) return;
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.removeAttribute('aria-disabled');
      link.classList.remove('disabled');
      link.querySelector('small').textContent = 'Open resource ↗';
    } catch { /* Keep invalid configuration safely disabled. */ }
  });
  if (config.correspondingEmail && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(config.correspondingEmail)) {
    const email = document.createElement('a');
    email.href = `mailto:${config.correspondingEmail}`;
    email.textContent = config.correspondingEmail;
    document.getElementById('contact').append(email);
  }
  const toggle = document.getElementById('animation-toggle');
  let paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const images = [...document.querySelectorAll('[data-rollout]')];
  const visible = new Set();
  function renderMedia() {
    images.forEach(img => {
      const play = visible.has(img) && !paused && !document.hidden;
      const src = 'assets/media/' + img.dataset.rollout + (play ? '.gif' : '.jpg');
      if (img.getAttribute('src') !== src) img.src = src;
    });
    toggle.textContent = paused ? 'Play animations' : 'Pause animations';
    toggle.setAttribute('aria-pressed', String(paused));
  }
  toggle.addEventListener('click', () => { paused = !paused; renderMedia(); });
  document.addEventListener('visibilitychange', renderMedia);
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => entry.isIntersecting ? visible.add(entry.target) : visible.delete(entry.target));
      renderMedia();
    }, { rootMargin: '80px' });
    images.forEach(img => observer.observe(img));
  } else { images.forEach(img => visible.add(img)); }
  renderMedia();
})();

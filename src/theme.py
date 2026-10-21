"""Starry-night look for the Gradio app: theme variables, CSS and a force-dark snippet.

Pure presentation. The star field is generated from a fixed seed (so it is identical on
every load) and sized in vw/vh, so it fills any screen. Animation is switched off for
visitors who ask for reduced motion.
"""

import random

import gradio as gr

PALETTE = {
    "night_0": "#05070f",
    "night_1": "#0b1226",
    "panel": "rgba(14, 20, 42, 0.74)",
    "panel_solid": "#0e142a",
    "line": "rgba(169, 199, 255, 0.20)",
    "text": "#e8eefc",
    "muted": "#aab6d3",
    "gold": "#f5c96a",
    "ice": "#a9c7ff",
}


def stars(count: int, seed: int) -> str:
    """A box-shadow list of tiny stars at random viewport positions."""
    rnd = random.Random(seed)
    tints = ["#ffffff", "#ffffff", "#cfe3ff", "#ffe9b8"]
    return ",".join(
        f"{rnd.uniform(0, 100):.1f}vw {rnd.uniform(0, 100):.1f}vh {rnd.choice(tints)}"
        for _ in range(count)
    )


# Always dark: a night sky has no light mode.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def build_theme() -> gr.themes.Base:
    p = PALETTE
    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="indigo",
        neutral_hue="slate",
        # System fonts only: no request to a font service from visitors' browsers.
        font=[gr.themes.Font(name) for name in ("ui-sans-serif", "system-ui", "Segoe UI", "sans-serif")],
    ).set(
        body_background_fill_dark="transparent",
        body_text_color_dark=p["text"],
        body_text_color_subdued_dark=p["muted"],
        background_fill_primary_dark=p["panel_solid"],
        background_fill_secondary_dark=p["panel"],
        block_background_fill_dark=p["panel"],
        block_border_color_dark=p["line"],
        block_label_background_fill_dark="rgba(22, 33, 63, 0.9)",
        block_label_text_color_dark=p["ice"],
        block_title_text_color_dark=p["ice"],
        panel_background_fill_dark=p["panel"],
        border_color_primary_dark=p["line"],
        input_background_fill_dark="#0a1126",
        input_border_color_dark=p["line"],
        input_placeholder_color_dark="#7d8bb0",
        button_primary_background_fill_dark=f"linear-gradient(135deg, {p['gold']}, #e0a43a)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #ffdb8a, #f0b24a)",
        button_primary_text_color_dark="#1a1200",
        button_secondary_background_fill_dark="rgba(34, 48, 92, 0.85)",
        button_secondary_background_fill_hover_dark="rgba(52, 70, 128, 0.95)",
        button_secondary_text_color_dark=p["text"],
        link_text_color_dark=p["gold"],
        link_text_color_hover_dark="#ffdb8a",
        color_accent_soft_dark="rgba(245, 201, 106, 0.16)",
    )


def build_css() -> str:
    p = PALETTE
    return f"""
body, .gradio-container {{
  background:
    radial-gradient(1100px 560px at 15% -8%, #1d2b5a 0%, transparent 62%),
    radial-gradient(900px 520px at 92% 6%, #3a2a6b 0%, transparent 58%),
    linear-gradient(180deg, {p['night_0']} 0%, {p['night_1']} 55%, #10182f 100%) !important;
  background-attachment: fixed !important;
}}

/* Two star layers with different twinkle timing, behind everything. */
.gradio-container::before, .gradio-container::after {{
  content: ""; position: fixed; top: 0; left: 0; width: 2px; height: 2px;
  border-radius: 50%; pointer-events: none; z-index: 0;
}}
.gradio-container::before {{ box-shadow: {stars(110, 7)}; animation: twinkle 5s ease-in-out infinite; }}
.gradio-container::after  {{ box-shadow: {stars(70, 19)}; animation: twinkle 8s ease-in-out infinite 2.5s; }}
.gradio-container > * {{ position: relative; z-index: 1; }}

/* An occasional shooting star. */
body::before {{
  content: ""; position: fixed; top: 12vh; left: -12vw; width: 130px; height: 2px;
  background: linear-gradient(90deg, transparent, #fff); border-radius: 2px;
  transform: rotate(18deg); opacity: 0; pointer-events: none; z-index: 0;
  animation: shoot 13s linear infinite 4s;
}}

@keyframes twinkle {{ 0%, 100% {{ opacity: .35; }} 50% {{ opacity: 1; }} }}
@keyframes shoot {{
  0% {{ transform: translate3d(0, 0, 0) rotate(18deg); opacity: 0; }}
  2% {{ opacity: 1; }}
  9% {{ transform: translate3d(75vw, 24vh, 0) rotate(18deg); opacity: 0; }}
  100% {{ transform: translate3d(75vw, 24vh, 0) rotate(18deg); opacity: 0; }}
}}
@keyframes glint {{ 0%, 100% {{ opacity: .55; transform: scale(.9); }} 50% {{ opacity: 1; transform: scale(1.15); }} }}
@keyframes sweep {{ from {{ background-position: 0 0; }} to {{ background-position: 200% 0; }} }}

/* Soft glass panels so the sky shows through the gaps. */
.block, .form {{ backdrop-filter: blur(6px); }}
.prose h1 {{ color: {p['text']}; letter-spacing: .02em; }}
.star-mark {{ display: inline-block; color: {p['gold']}; animation: glint 3.2s ease-in-out infinite; }}

/* Loading bar for indexing: gold-to-ice, with a moving sheen so it never looks frozen. */
.progress-bar, .eta-bar {{
  background: linear-gradient(90deg, {p['gold']}, {p['ice']}, {p['gold']}) !important;
  background-size: 200% 100% !important; animation: sweep 2.4s linear infinite;
}}
.progress-text, .progress-level, .meta-text, .meta-text-center {{ color: {p['text']} !important; }}

@media (prefers-reduced-motion: reduce) {{
  .gradio-container::before, .gradio-container::after, body::before,
  .star-mark, .progress-bar, .eta-bar {{ animation: none !important; }}
}}
"""
